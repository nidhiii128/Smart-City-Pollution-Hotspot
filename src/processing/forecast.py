"""Stage 4 - air-quality forecasting with Spark MLlib.

For every site a Gradient-Boosted Tree regressor predicts the next day's mean
value of each target (PM2.5 and AQI) from:
  * autoregressive lags (t, t-1, t-2, t-6), the 7-day rolling mean and
    today's deviation from it,
  * the weather of the day being predicted (Open-Meteo forecast when
    available, otherwise site climatology for that calendar month),
  * day of week.
The model is deliberately small (depth-2 trees, strong regularisation): on
real CPCB data, larger models with site and seasonal features overfit and
lost to persistence.

Evaluation uses a chronological hold-out (last 20% of days) and is compared
against a persistence baseline (tomorrow = today). The final model is refit
on all data and rolled forward recursively for `horizon_days`.
"""
from __future__ import annotations

import json
import logging
import math
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from src.aqi import category
from src.config import load_config

log = logging.getLogger("forecast")

WEATHER = ["temperature", "humidity", "wind_speed", "precipitation", "pressure"]
LAGS = [0, 1, 2, 6]


MODEL_WEATHER = ["wind_speed", "precipitation", "humidity", "temperature"]


def feature_names(target: str) -> list[str]:
    return ([f"{target}_lag{l}" for l in LAGS] + [f"{target}_roll7", f"{target}_dev7", "dow"]
            + [f"next_{w}" for w in MODEL_WEATHER])


def build_training_set(daily: DataFrame, target: str) -> DataFrame:
    w = Window.partitionBy("site_id").orderBy("date")
    df = daily.select("site_id", "date", "lat", "lon", "zone", target, *WEATHER)
    cols = {f"{target}_lag{l}": F.lag(target, l).over(w) if l else F.col(target) for l in LAGS}
    cols[f"{target}_roll7"] = F.avg(target).over(w.rowsBetween(-6, 0))
    # The model predicts the day-to-day CHANGE; the forecast is today's value + change.
    # Trees cannot extrapolate beyond the label range seen in training, so learning
    # the change keeps working when pollution levels shift (e.g. monsoon -> winter).
    cols["label_level"] = F.lead(target, 1).over(w)
    cols["label"] = F.lead(target, 1).over(w) - F.col(target)
    cols["label_date"] = F.lead("date", 1).over(w)
    cols.update({f"next_{x}": F.lead(x, 1).over(w) for x in WEATHER})
    df = df.withColumns(cols)
    df = df.filter(F.datediff("label_date", "date") == 1)  # skip gaps in the series
    df = df.withColumns({
        f"{target}_dev7": F.col(f"{target}_lag0") - F.col(f"{target}_roll7"),
        "dow": F.dayofweek("label_date").cast("double"),
    })
    return df.dropna(subset=feature_names(target) + ["label"])


def make_pipeline(target: str, seed: int) -> Pipeline:
    return Pipeline(stages=[
        VectorAssembler(inputCols=feature_names(target), outputCol="features"),
        GBTRegressor(featuresCol="features", labelCol="label", predictionCol="prediction",
                     maxIter=60, maxDepth=2, stepSize=0.05, minInstancesPerNode=30,
                     subsamplingRate=0.8, seed=seed),
    ])


def evaluate(pred: DataFrame, col: str = "prediction") -> dict[str, float]:
    out = {}
    for metric in ("rmse", "mae", "r2"):
        ev = RegressionEvaluator(labelCol="label_level", predictionCol=col, metricName=metric)
        out[metric] = round(ev.evaluate(pred), 3)
    return out


def train_and_evaluate(spark: SparkSession, daily: DataFrame, target: str,
                       test_fraction: float, seed: int) -> tuple[PipelineModel, dict, pd.DataFrame, DataFrame]:
    data = build_training_set(daily, target).cache()
    dates = sorted(r["label_date"] for r in data.select("label_date").distinct().collect())
    split_date = dates[int(len(dates) * (1 - test_fraction))]
    train = data.filter(F.col("label_date") < split_date)
    test = data.filter(F.col("label_date") >= split_date)

    model = make_pipeline(target, seed).fit(train)
    pred = (model.transform(test)
            .withColumn("pred_level", F.greatest(F.lit(0.0), F.col(f"{target}_lag0") + F.col("prediction")))
            .withColumn("baseline", F.col(f"{target}_lag0")))
    metrics = {
        "target": target,
        "model": "GBTRegressor",
        "train_rows": train.count(), "test_rows": test.count(),
        "split_date": str(split_date),
        **evaluate(pred, "pred_level"),
        **{f"baseline_{k}": v for k, v in evaluate(pred, "baseline").items()},
    }
    metrics["improvement_vs_baseline_pct"] = round(
        100 * (metrics["baseline_rmse"] - metrics["rmse"]) / metrics["baseline_rmse"], 1)

    gbt = model.stages[-1]
    importance = pd.DataFrame({"target": target, "feature": feature_names(target),
                               "importance": gbt.featureImportances.toArray().round(4)})
    test_pred = pred.select("site_id", F.col("label_date").alias("date"),
                            F.lit(target).alias("target"), F.col("label_level").alias("actual"),
                            F.round("pred_level", 2).alias("predicted"),
                            F.round("baseline", 2).alias("baseline"))
    log.info("%s model: %s", target, metrics)

    final_model = make_pipeline(target, seed).fit(data)  # refit on everything for the forecast
    data.unpersist()
    return final_model, metrics, importance.sort_values("importance", ascending=False), test_pred


def future_weather(daily_pd: pd.DataFrame, raw_dir: Path, last_date: pd.Timestamp,
                   horizon: int) -> pd.DataFrame:
    """Daily weather per site for the forecast horizon (forecast API or climatology)."""
    days = [last_date + timedelta(days=h) for h in range(1, horizon + 1)]
    fc_path = raw_dir / "weather_forecast.parquet"
    if fc_path.exists():
        fc = pd.read_parquet(fc_path)
        fc["date"] = pd.to_datetime(fc["timestamp"]).dt.normalize()
        agg = {w: "mean" for w in WEATHER}
        agg["precipitation"] = "sum"
        wx = fc.groupby(["site_id", "date"]).agg(agg).reset_index()
        wx["weather_source"] = "Open-Meteo forecast"
    else:
        wx = pd.DataFrame()
    clim = (daily_pd.assign(m=daily_pd["date"].dt.month)
            .groupby(["site_id", "m"])[WEATHER].mean().reset_index())
    rows = []
    for site in daily_pd["site_id"].unique():
        for d in days:
            hit = wx[(wx["site_id"] == site) & (wx["date"] == d)] if len(wx) else wx
            if len(hit):
                rows.append(hit.iloc[0].to_dict())
            else:
                c = clim[(clim["site_id"] == site) & (clim["m"] == d.month)]
                if c.empty:
                    c = clim[clim["site_id"] == site]
                rows.append({"site_id": site, "date": d, **c[WEATHER].mean().to_dict(),
                             "weather_source": "climatology"})
    return pd.DataFrame(rows)


def forecastable_sites(daily_pd: pd.DataFrame, max_staleness_days: int = 3) -> pd.DataFrame:
    """Anchor the forecast on the latest day most sites have reported.

    Ground-station feeds arrive with a lag that differs per station, so the
    single newest day is often covered by only one or two sites. Days after the
    anchor are dropped, as are sites whose data stops well before it.
    """
    n_sites = daily_pd["site_id"].nunique()
    per_day = daily_pd.groupby("date")["site_id"].nunique()
    anchor = per_day[per_day >= 0.5 * n_sites].index.max()
    out = daily_pd[daily_pd["date"] <= anchor]
    last_seen = out.groupby("site_id")["date"].max()
    lens = out.groupby("site_id").size()
    keep = last_seen[(last_seen >= anchor - timedelta(days=max_staleness_days))
                     & (lens.reindex(last_seen.index) >= max(LAGS) + 1)].index
    dropped = n_sites - len(keep)
    log.info("Forecast anchored on %s (%d sites%s)", anchor.date(), len(keep),
             f", {dropped} with stale or short history skipped" if dropped else "")
    return out[out["site_id"].isin(keep)]


def persistence_forecast(daily_pd: pd.DataFrame, target: str, horizon: int,
                         rmse: float) -> pd.DataFrame:
    """Fallback forecaster: every future day equals the last observed day."""
    last = daily_pd.sort_values("date").groupby("site_id").tail(1)
    anchor = daily_pd["date"].max()
    out = []
    for _, r in last.iterrows():
        for h in range(1, horizon + 1):
            band = 1.2816 * rmse * math.sqrt(h)
            out.append({"site_id": r["site_id"], "date": anchor + timedelta(days=h), "horizon_day": h,
                        "target": target, "predicted": round(float(r[target]), 2),
                        "lower_80": round(max(0.0, r[target] - band), 2),
                        "upper_80": round(r[target] + band, 2)})
    return pd.DataFrame(out)


def recursive_forecast(spark: SparkSession, model: PipelineModel, daily_pd: pd.DataFrame,
                       weather: pd.DataFrame, target: str, horizon: int, rmse: float) -> pd.DataFrame:
    history = {s: g.sort_values("date")[target].tolist() for s, g in daily_pd.groupby("site_id")}
    meta = daily_pd.drop_duplicates("site_id").set_index("site_id")[["lat", "lon", "zone"]]
    last_date = daily_pd["date"].max()
    out = []
    for h in range(1, horizon + 1):
        d = last_date + timedelta(days=h)
        rows = []
        for site, hist in history.items():
            wx = weather[(weather["site_id"] == site) & (weather["date"] == d)].iloc[0]
            row = {"site_id": site, "zone": meta.loc[site, "zone"],
                   "lat": float(meta.loc[site, "lat"]), "lon": float(meta.loc[site, "lon"]),
                   f"{target}_roll7": float(np.mean(hist[-7:])),
                   f"{target}_dev7": float(hist[-1] - np.mean(hist[-7:])),
                   "dow": float(d.dayofweek + 2 if d.dayofweek < 6 else 1)}  # Spark: Sun=1
            row.update({f"{target}_lag{l}": float(hist[-1 - l]) for l in LAGS})
            row.update({f"next_{w}": float(wx[w]) for w in WEATHER})
            rows.append(row)
        pred = model.transform(spark.createDataFrame(pd.DataFrame(rows))) \
                    .select("site_id", "prediction").toPandas()
        for _, r in pred.iterrows():
            value = max(0.0, history[r["site_id"]][-1] + float(r["prediction"]))
            history[r["site_id"]].append(value)
            band = 1.2816 * rmse * math.sqrt(h)  # ~80% interval, widening with horizon
            out.append({"site_id": r["site_id"], "date": d, "horizon_day": h, "target": target,
                        "predicted": round(value, 2),
                        "lower_80": round(max(0.0, value - band), 2),
                        "upper_80": round(value + band, 2)})
    return pd.DataFrame(out)


def run(spark: SparkSession) -> dict:
    cfg = load_config()
    pcfg = cfg["prediction"]
    out_dir, raw_dir = Path(cfg["paths"]["processed"]), Path(cfg["paths"]["raw"])
    model_dir = Path(cfg["paths"]["models"])

    daily = spark.read.parquet(str(out_dir / "daily_site"))
    daily_pd = daily.select("site_id", "site_name", "city", "date", "lat", "lon", "zone",
                            *pcfg["targets"], *WEATHER).toPandas()
    daily_pd["date"] = pd.to_datetime(daily_pd["date"])
    daily_pd = forecastable_sites(daily_pd)
    last_date = daily_pd["date"].max()
    weather = future_weather(daily_pd, raw_dir, last_date, pcfg["horizon_days"])

    metrics, importances, tests, forecasts = [], [], [], []
    for target in pcfg["targets"]:
        model, m, imp, test_pred = train_and_evaluate(spark, daily, target,
                                                      pcfg["test_fraction"], pcfg["seed"])
        model.write().overwrite().save(str(model_dir / f"gbt_{target}"))
        metrics.append(m)
        importances.append(imp)
        tests.append(test_pred.toPandas())
        # model selection: only use the GBT forecast if it beat persistence on the hold-out
        if m["rmse"] <= m["baseline_rmse"]:
            m["selected_method"] = "GBTRegressor"
            forecasts.append(recursive_forecast(spark, model, daily_pd, weather, target,
                                                pcfg["horizon_days"], m["rmse"]))
        else:
            m["selected_method"] = "Persistence"
            log.warning("%s: GBT (RMSE %.1f) did not beat persistence (RMSE %.1f) on the hold-out "
                        "- forecasting with persistence. More history usually fixes this.",
                        target, m["rmse"], m["baseline_rmse"])
            forecasts.append(persistence_forecast(daily_pd, target, pcfg["horizon_days"],
                                                  m["baseline_rmse"]))

    fc = pd.concat(forecasts, ignore_index=True)
    wide = fc.pivot_table(index=["site_id", "date", "horizon_day"], columns="target",
                          values=["predicted", "lower_80", "upper_80"]).reset_index()
    wide.columns = ["_".join(c for c in col if c) if isinstance(col, tuple) else col
                    for col in wide.columns]
    wide = wide.rename(columns={f"predicted_{t}": f"pred_{t}" for t in pcfg["targets"]})
    if "pred_aqi" in wide:
        wide["pred_aqi_category"] = wide["pred_aqi"].map(category)
    names = daily_pd.drop_duplicates("site_id").set_index("site_id")
    wide = wide.join(names[["site_name", "city", "lat", "lon"]], on="site_id")
    wide = wide.merge(weather[["site_id", "date", "weather_source"]], on=["site_id", "date"], how="left")

    test_df = pd.concat(tests, ignore_index=True)
    test_df["date"] = pd.to_datetime(test_df["date"])
    for name, frame in {"forecasts": wide, "model_metrics": pd.DataFrame(metrics),
                        "feature_importance": pd.concat(importances, ignore_index=True),
                        "test_predictions": test_df}.items():
        frame.to_parquet(out_dir / f"{name}.parquet", index=False, coerce_timestamps="us")

    (out_dir / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("Forecast generated for %s -> %s", wide["date"].min().date(), wide["date"].max().date())
    return {"metrics": metrics, "forecast_days": pcfg["horizon_days"]}
