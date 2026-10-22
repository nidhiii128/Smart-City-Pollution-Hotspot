"""Stage 3 - pollution hotspot detection.

Two complementary methods:

1. K-Means clustering (Spark MLlib) on standardised per-site pollution
   profiles. k is chosen automatically by silhouette score and clusters are
   ranked by their mean AQI to give human-readable severity labels.
2. Getis-Ord Gi* spatial statistic on mean PM2.5. It flags sites that are
   high AND surrounded by other high sites - statistically significant
   spatial hotspots rather than isolated bad readings.

Clustering is also repeated per season so the dashboard can show how
hotspots move through the year.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.config import load_config

log = logging.getLogger("hotspots")

# Clustering uses particulate + AQI features: they drive the AQI and every station
# measures them. Gas sensors are missing at many CPCB stations, and imputing them
# made clusters reflect instrument coverage rather than pollution severity.
FEATURES = ["mean_pm2_5", "mean_pm10", "p95_pm2_5", "pct_days_pm25_exceed", "mean_aqi"]
# reported per cluster for interpretation
PROFILE_FEATURES = ["mean_pm2_5", "mean_pm10", "mean_no2", "mean_so2", "mean_co", "mean_o3",
                    "p95_pm2_5", "pct_days_pm25_exceed", "mean_aqi"]
SEVERITY = ["Critical Hotspot", "Hotspot", "Elevated", "Moderate", "Low Pollution", "Clean Zone"]
GI_BAND_KM = 8.0


def severity_labels(k: int) -> list[str]:
    """Rank 0 = worst cluster. Always keeps the extremes, fills the middle."""
    if k >= len(SEVERITY):
        return SEVERITY + [f"Level {i}" for i in range(len(SEVERITY), k)]
    middle = SEVERITY[1:-1][: k - 2]
    return [SEVERITY[0], *middle, SEVERITY[-2] if k > 2 else SEVERITY[-1]]


def kmeans_best(df: DataFrame, features: list[str], k_min: int, k_max: int,
                seed: int) -> tuple[DataFrame, int, dict[int, float]]:
    # stations that don't measure a pollutant get the network mean, so they still get clustered
    means = df.select(*[F.avg(c).alias(c) for c in features]).first().asDict()
    df = df.fillna({c: (v if v is not None else 0.0) for c, v in means.items()})
    assembler = VectorAssembler(inputCols=features, outputCol="raw_features", handleInvalid="skip")
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True, withStd=True)
    evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="cluster",
                                    metricName="silhouette")
    n = df.count()
    scores, best = {}, None
    for k in range(k_min, min(k_max, n - 1) + 1):
        km = KMeans(k=k, seed=seed, featuresCol="features", predictionCol="cluster", maxIter=50)
        model = Pipeline(stages=[assembler, scaler, km]).fit(df)
        pred = model.transform(df)
        scores[k] = round(evaluator.evaluate(pred), 4)
        if best is None or scores[k] > scores[best[0]]:
            best = (k, pred)
    log.info("Silhouette by k: %s -> best k=%d", scores, best[0])
    return best[1], best[0], scores


def label_clusters(pred: DataFrame, k: int) -> DataFrame:
    order = (pred.groupBy("cluster").agg(F.avg("mean_aqi").alias("c_aqi"))
             .orderBy(F.desc("c_aqi")).collect())
    labels = severity_labels(k)
    mapping = {row["cluster"]: (rank, labels[rank]) for rank, row in enumerate(order)}
    rank_map = F.create_map(*[x for c, (r, _) in mapping.items() for x in (F.lit(c), F.lit(r))])
    label_map = F.create_map(*[x for c, (_, l) in mapping.items() for x in (F.lit(c), F.lit(l))])
    return (pred.withColumn("severity_rank", rank_map[F.col("cluster")])
                .withColumn("hotspot_label", label_map[F.col("cluster")]))


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def getis_ord_gi_star(sites: pd.DataFrame, value: str, band_km: float) -> pd.DataFrame:
    lat, lon, x = sites["lat"].to_numpy(), sites["lon"].to_numpy(), sites[value].to_numpy()
    d = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    w = (d <= band_km).astype(float)  # binary distance band, includes self (Gi*)
    n, xbar = len(x), x.mean()
    s = math.sqrt((x ** 2).mean() - xbar ** 2)
    wsum = w.sum(1)
    num = w @ x - xbar * wsum
    den = s * np.sqrt((n * (w ** 2).sum(1) - wsum ** 2) / (n - 1))
    z = np.where(den > 0, num / den, 0.0)
    out = sites[["site_id"]].copy()
    out["gi_z"] = np.round(z, 3)
    out["gi_neighbours"] = (wsum - 1).astype(int)
    out["gi_class"] = np.select(
        [z >= 2.576, z >= 1.96, z >= 1.645, z <= -2.576, z <= -1.96, z <= -1.645],
        ["Hotspot (99%)", "Hotspot (95%)", "Hotspot (90%)",
         "Coldspot (99%)", "Coldspot (95%)", "Coldspot (90%)"], "Not significant")
    return out


def seasonal_hotspots(spark: SparkSession, daily: DataFrame, cfg: dict) -> DataFrame:
    feats = ["mean_pm2_5", "mean_pm10", "mean_aqi"]
    seasonal = (daily.groupBy("site_id", "site_name", "city", "lat", "lon", "season")
                .agg(F.avg("pm2_5").alias("mean_pm2_5"), F.avg("pm10").alias("mean_pm10"),
                     F.avg("no2").alias("mean_no2"), F.avg("so2").alias("mean_so2"),
                     F.avg("aqi").alias("mean_aqi")))
    results = []
    for season in [r["season"] for r in seasonal.select("season").distinct().collect()]:
        part = seasonal.filter(F.col("season") == season)
        if part.count() < 5:
            continue
        pred, k, _ = kmeans_best(part, feats, 3, 4, cfg["clustering"]["seed"])
        results.append(label_clusters(pred, k)
                       .select("site_id", "site_name", "city", "lat", "lon", "season",
                               *[F.round(c, 2).alias(c) for c in feats],
                               "cluster", "severity_rank", "hotspot_label"))
    out = results[0]
    for r in results[1:]:
        out = out.unionByName(r)
    return out


def run(spark: SparkSession) -> dict:
    cfg = load_config()
    out_dir = Path(cfg["paths"]["processed"])
    summary = spark.read.parquet(str(out_dir / "site_summary"))
    daily = spark.read.parquet(str(out_dir / "daily_site"))

    pred, k, scores = kmeans_best(summary, FEATURES, cfg["clustering"]["k_min"],
                                  cfg["clustering"]["k_max"], cfg["clustering"]["seed"])
    labelled = label_clusters(pred, k).drop("raw_features", "features").toPandas()

    gi = getis_ord_gi_star(labelled, "mean_pm2_5", GI_BAND_KM)
    hot = labelled.merge(gi, on="site_id")
    # composite 0-100 score: AQI level, exceedance frequency and spatial clustering
    z = lambda s: (s - s.mean()) / (s.std() or 1)  # noqa: E731
    raw_score = z(hot["mean_aqi"]) + z(hot["pct_days_pm25_exceed"]) + 0.5 * z(hot["gi_z"])
    hot["hotspot_score"] = (100 * (raw_score - raw_score.min())
                            / (raw_score.max() - raw_score.min())).round(1)
    hot = hot.sort_values("hotspot_score", ascending=False).reset_index(drop=True)
    hot["hotspot_rank"] = hot.index + 1

    spark.createDataFrame(hot).coalesce(1).write.mode("overwrite").parquet(str(out_dir / "hotspots"))

    centers = (hot.groupby(["cluster", "hotspot_label", "severity_rank"])[PROFILE_FEATURES]
               .mean().round(2).reset_index())
    centers["sites"] = hot.groupby("cluster").size().reindex(centers["cluster"]).to_numpy()
    spark.createDataFrame(centers).coalesce(1).write.mode("overwrite") \
        .parquet(str(out_dir / "cluster_profiles"))

    seasonal_hotspots(spark, daily, cfg).coalesce(1).write.mode("overwrite") \
        .parquet(str(out_dir / "seasonal_hotspots"))

    info = {"k": k, "silhouette": scores,
            "critical_sites": hot.loc[hot["severity_rank"] == 0, "site_name"].tolist(),
            "gi_hotspots": hot.loc[hot["gi_class"].str.startswith("Hotspot"), "site_name"].tolist()}
    pd.Series({"k": k, "silhouette": scores[k]}).to_json(out_dir / "clustering_meta.json")
    log.info("Hotspots: %s", info)
    return info
