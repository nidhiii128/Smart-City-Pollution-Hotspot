"""Stage 2 - Spark ETL.

raw hourly parquet  ->  cleaned hourly data with rolling averages + CPCB AQI
                    ->  daily / monthly / diurnal / city aggregates
                    ->  per-site summary features (input to hotspot clustering)
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from src.aqi import BREAKPOINTS, CATEGORIES, NAAQS_24H
from src.config import load_config

log = logging.getLogger("etl")

POLLUTANTS = ["pm2_5", "pm10", "no2", "so2", "co", "o3"]
WEATHER = ["temperature", "humidity", "wind_speed", "wind_direction", "precipitation", "pressure"]
# physically plausible upper limits (ug/m3); anything above is a sensor fault
# PM limits sit just below the ~1000 ug/m3 full scale of CPCB beta-attenuation monitors:
# readings at the ceiling (985, 1000, ...) are saturation / error codes, not measurements.
MAX_VALID = {"pm2_5": 950, "pm10": 1100, "no2": 1000, "so2": 2000, "co": 50000, "o3": 1000}
# averaging window (hours) used by CPCB for each pollutant's sub-index
AVG_HOURS = {"pm2_5": 24, "pm10": 24, "no2": 24, "so2": 24, "co": 8, "o3": 8}
MAX_FILL_HOURS = 3
# identical readings for this many consecutive hours = stuck sensor
FLATLINE_HOURS = 12
# spike = more than SPIKE_RATIO x the local median AND at least this many ug/m3 above it
SPIKE_RATIO = 4.0
SPIKE_MIN_JUMP = {"pm2_5": 100, "pm10": 150, "no2": 100, "so2": 100, "co": 3000, "o3": 100}
# CPCB: a daily value needs at least 16 valid hours
MIN_HOURS_PER_DAY = 16


def sub_index_expr(pollutant: str, col: F.Column) -> F.Column:
    """Piecewise-linear CPCB sub-index as a native Spark expression."""
    expr, prev_hi = None, 0.0
    for _, hi, ilo, ihi in BREAKPOINTS[pollutant]:
        branch = F.lit(ilo) + F.lit(ihi - ilo) * (col - F.lit(prev_hi)) / F.lit(hi - prev_hi)
        expr = F.when(col <= hi, branch) if expr is None else expr.when(col <= hi, branch)
        prev_hi = hi
    return F.when(col.isNull(), None).otherwise(expr.otherwise(F.lit(500.0)))


def category_expr(aqi: F.Column) -> F.Column:
    expr = None
    for upper, name, _ in CATEGORIES:
        expr = F.when(aqi <= upper, name) if expr is None else expr.when(aqi <= upper, name)
    return F.when(aqi.isNull(), "Unknown").otherwise(expr)


def season_expr(month: F.Column) -> F.Column:
    return (F.when(month.isin(12, 1, 2), "Winter")
             .when(month.isin(3, 4, 5), "Summer")
             .when(month.isin(6, 7, 8, 9), "Monsoon")
             .otherwise("Post-Monsoon"))


def load_sites(spark: SparkSession, raw_dir: Path) -> DataFrame:
    sites = json.loads((raw_dir / "sites.json").read_text())
    rows = [(s["id"], s["name"], s["city"], float(s["lat"]), float(s["lon"]), s.get("zone", "unknown"))
            for s in sites]
    return spark.createDataFrame(rows, "site_id string, site_name string, city string, "
                                       "lat double, lon double, zone string")


def clean_hourly(spark: SparkSession, raw_dir: Path) -> DataFrame:
    raw = spark.read.parquet(str(raw_dir / "hourly"))

    # Timestamps are local IST wall-clock times with no zone; keep them zone-less
    # (TIMESTAMP_NTZ) so they are written back unchanged instead of shifted to UTC.
    df = raw.withColumns({"timestamp": F.col("timestamp").cast("timestamp_ntz"),
                          **{p: F.col(p).cast("double") for p in POLLUTANTS + WEATHER}})
    df = df.dropDuplicates(["site_id", "timestamp"])

    # 1. invalid readings -> null
    df = df.withColumns({p: F.when((F.col(p) < 0) | (F.col(p) > MAX_VALID[p]), None)
                              .otherwise(F.col(p)) for p in POLLUTANTS})

    # 1b. remove flatlines: a sensor reporting exactly the same value for many
    #     consecutive hours is stuck (common in CPCB feeds, e.g. PM2.5 = 985 for days)
    w_ord = Window.partitionBy("site_id").orderBy("timestamp")
    df = df.withColumns({f"{p}_chg": F.when(F.col(p).eqNullSafe(F.lag(p).over(w_ord)), 0).otherwise(1)
                         for p in POLLUTANTS})
    df = df.withColumns({f"{p}_run": F.sum(f"{p}_chg").over(w_ord.rowsBetween(Window.unboundedPreceding, 0))
                         for p in POLLUTANTS})
    df = df.withColumns({f"{p}_runlen": F.count("*").over(Window.partitionBy("site_id", f"{p}_run"))
                         for p in POLLUTANTS})
    df = df.withColumns({p: F.when(F.col(f"{p}_runlen") >= FLATLINE_HOURS, None).otherwise(F.col(p))
                         for p in POLLUTANTS})
    df = df.drop(*[f"{p}_{x}" for p in POLLUTANTS for x in ("chg", "run", "runlen")])

    # 2a. remove isolated sensor spikes (Hampel-style): a reading far above the
    #     rolling median of the surrounding +/-12 hours is treated as a fault
    w_med = Window.partitionBy("site_id").orderBy("timestamp").rowsBetween(-12, 12)
    df = df.withColumns({f"{p}_med": F.percentile_approx(p, 0.5).over(w_med) for p in POLLUTANTS})
    df = df.withColumns({p: F.when((F.col(p) > SPIKE_RATIO * F.col(f"{p}_med"))
                                   & (F.col(p) - F.col(f"{p}_med") > SPIKE_MIN_JUMP[p]), None)
                              .otherwise(F.col(p)) for p in POLLUTANTS})
    df = df.drop(*[f"{p}_med" for p in POLLUTANTS])

    # 2b. cap remaining statistical outliers at the per-site 99.9th percentile
    caps = df.groupBy("site_id").agg(*[F.percentile_approx(p, 0.999).alias(f"{p}_cap")
                                       for p in POLLUTANTS])
    df = (df.join(F.broadcast(caps), "site_id")
            # note: F.least() skips nulls, so missing readings must stay null explicitly
            .withColumns({p: F.when(F.col(p).isNull(), None)
                              .otherwise(F.least(F.col(p), F.col(f"{p}_cap"))) for p in POLLUTANTS})
            .drop(*[f"{p}_cap" for p in POLLUTANTS]))

    # 3. forward-fill short gaps (<= 3h) per site
    w_fill = (Window.partitionBy("site_id").orderBy("timestamp")
              .rowsBetween(-MAX_FILL_HOURS, 0))
    df = df.withColumns({p: F.coalesce(F.col(p), F.last(p, ignorenulls=True).over(w_fill))
                         for p in POLLUTANTS + WEATHER})

    # 4. CPCB rolling averages (24h / 8h) and sub-indices
    df = df.withColumn("ts_epoch", F.unix_timestamp(F.col("timestamp").cast("string")))
    avgs = {}
    for p in POLLUTANTS:
        w = (Window.partitionBy("site_id").orderBy("ts_epoch")
             .rangeBetween(-(AVG_HOURS[p] - 1) * 3600, 0))
        avgs[f"{p}_avg"] = F.avg(p).over(w) / (1000.0 if p == "co" else 1.0)  # CO index in mg/m3
    df = df.withColumns(avgs)
    df = df.withColumns({f"si_{p}": sub_index_expr(p, F.col(f"{p}_avg")) for p in POLLUTANTS})
    df = df.withColumn("aqi_raw", F.greatest(*[F.col(f"si_{p}") for p in POLLUTANTS]))
    dominant = None
    for p in POLLUTANTS:
        cond = F.col(f"si_{p}") == F.col("aqi_raw")
        dominant = F.when(cond, p) if dominant is None else dominant.when(cond, p)
    # CPCB requires at least one particulate sub-index
    has_pm = F.col("si_pm2_5").isNotNull() | F.col("si_pm10").isNotNull()
    df = df.withColumns({"aqi": F.when(has_pm, F.round("aqi_raw", 1)),
                         "dominant_pollutant": dominant})
    df = (df.withColumn("aqi_category", category_expr(F.col("aqi")))
            .drop("ts_epoch", "aqi_raw", *avgs))

    # 5. calendar features + site metadata
    df = df.withColumns({"date": F.to_date("timestamp"),
                         "hour": F.hour("timestamp"),
                         "month": F.date_format("timestamp", "yyyy-MM"),
                         "day_of_week": F.date_format("timestamp", "E"),
                         "season": season_expr(F.month("timestamp"))})
    df = df.join(F.broadcast(load_sites(spark, raw_dir)), "site_id", "left")
    df = df.dropna(subset=["aqi"])
    df = df.withColumns({c: F.round(c, 2) for c in
                         [f"si_{p}" for p in POLLUTANTS] + POLLUTANTS + WEATHER})
    log.info("Built cleaning plan for %s", raw_dir / "hourly")
    return df


def build_aggregates(hourly: DataFrame) -> dict[str, DataFrame]:
    meta = ["site_id", "site_name", "city", "lat", "lon", "zone"]
    pol_means = [F.round(F.avg(p), 2).alias(p) for p in POLLUTANTS]

    daily = (hourly.groupBy(*meta, "date", "season")
             .agg(*pol_means,
                  F.round(F.max("pm2_5"), 2).alias("pm2_5_max"),
                  F.round(F.avg("aqi"), 1).alias("aqi"),
                  F.round(F.max("aqi"), 1).alias("aqi_max"),
                  F.mode("dominant_pollutant").alias("dominant_pollutant"),
                  F.round(F.avg("temperature"), 2).alias("temperature"),
                  F.round(F.avg("humidity"), 2).alias("humidity"),
                  F.round(F.avg("wind_speed"), 2).alias("wind_speed"),
                  F.round(F.sum("precipitation"), 2).alias("precipitation"),
                  F.round(F.avg("pressure"), 2).alias("pressure"),
                  F.count("*").alias("hours"))
             .filter(F.col("hours") >= MIN_HOURS_PER_DAY)
             .withColumn("aqi_category", category_expr(F.col("aqi")))
             .withColumn("pm2_5_exceeds", F.col("pm2_5") > NAAQS_24H["pm2_5"]))

    monthly = (hourly.groupBy(*meta, "month")
               .agg(*pol_means, F.round(F.avg("aqi"), 1).alias("aqi"),
                    F.round(F.max("aqi"), 1).alias("aqi_max"))
               .withColumn("aqi_category", category_expr(F.col("aqi"))))

    diurnal = (hourly.groupBy("site_id", "site_name", "city", "hour")
               .agg(*pol_means, F.round(F.avg("aqi"), 1).alias("aqi")))

    city_daily = (hourly.groupBy("city", "date")
                  .agg(*pol_means, F.round(F.avg("aqi"), 1).alias("aqi"),
                       F.round(F.max("aqi"), 1).alias("aqi_max"))
                  .withColumn("aqi_category", category_expr(F.col("aqi"))))

    peak_hour = (diurnal.withColumn("rk", F.row_number().over(
                     Window.partitionBy("site_id").orderBy(F.desc("aqi"))))
                 .filter("rk = 1").select("site_id", F.col("hour").alias("peak_hour")))
    winter = (daily.filter(F.col("season").isin("Winter", "Post-Monsoon"))
              .groupBy("site_id").agg(F.round(F.avg("pm2_5"), 2).alias("winter_pm2_5")))
    summary = (daily.groupBy(*meta)
               .agg(*[F.round(F.avg(p), 2).alias(f"mean_{p}") for p in POLLUTANTS],
                    F.round(F.percentile_approx("pm2_5", 0.95), 2).alias("p95_pm2_5"),
                    F.round(F.avg("aqi"), 1).alias("mean_aqi"),
                    F.round(F.max("aqi_max"), 1).alias("max_aqi"),
                    F.round(100 * F.avg(F.col("pm2_5_exceeds").cast("int")), 1)
                     .alias("pct_days_pm25_exceed"),
                    F.round(100 * F.avg((F.col("aqi") > 200).cast("int")), 1)
                     .alias("pct_days_poor"),
                    F.mode("dominant_pollutant").alias("dominant_pollutant"),
                    F.count("*").alias("days"))
               .join(peak_hour, "site_id", "left")
               .join(winter, "site_id", "left")
               .withColumn("aqi_category", category_expr(F.col("mean_aqi"))))

    return {"daily_site": daily, "monthly_site": monthly, "diurnal_profile": diurnal,
            "city_daily": city_daily, "site_summary": summary}


def run(spark: SparkSession) -> dict[str, int]:
    t0 = time.time()
    cfg = load_config()
    raw_dir, out_dir = Path(cfg["paths"]["raw"]), Path(cfg["paths"]["processed"])

    hourly = clean_hourly(spark, raw_dir)
    hourly.repartition("month").write.mode("overwrite").partitionBy("month") \
        .parquet(str(out_dir / "hourly_clean"))
    hourly = spark.read.parquet(str(out_dir / "hourly_clean")).cache()

    counts = {"hourly_clean": hourly.count()}
    for name, df in build_aggregates(hourly).items():
        df.coalesce(1).write.mode("overwrite").parquet(str(out_dir / name))
        counts[name] = spark.read.parquet(str(out_dir / name)).count()
    hourly.unpersist()
    log.info("ETL finished in %.0fs, row counts: %s", time.time() - t0, counts)
    return counts
