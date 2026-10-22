"""Stage 5 - MongoDB serving layer.

Processed Spark outputs are loaded into MongoDB collections that the
dashboard queries. The large hourly table is written in parallel from Spark
executors (one bulk insert per partition); small analytic tables are written
from the driver.

Collections
-----------
sites, hourly_measurements, daily_stats, monthly_stats, diurnal_profile,
city_daily, hotspots, cluster_profiles, seasonal_hotspots, forecasts,
model_metrics, feature_importance, test_predictions, pipeline_runs
"""
from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pymongo import ASCENDING, DESCENDING, GEOSPHERE, MongoClient
from pymongo.errors import PyMongoError

from src.config import load_config

log = logging.getLogger("mongo")

# collection -> processed dataset (parquet file or Spark output directory)
DATASETS = {
    "daily_stats": "daily_site",
    "monthly_stats": "monthly_site",
    "diurnal_profile": "diurnal_profile",
    "city_daily": "city_daily",
    "hotspots": "hotspots",
    "cluster_profiles": "cluster_profiles",
    "seasonal_hotspots": "seasonal_hotspots",
    "forecasts": "forecasts.parquet",
    "model_metrics": "model_metrics.parquet",
    "feature_importance": "feature_importance.parquet",
    "test_predictions": "test_predictions.parquet",
}
INDEXES = {
    "hourly_measurements": [([("site_id", ASCENDING), ("timestamp", ASCENDING)], True),
                            ([("timestamp", DESCENDING)], False),
                            ([("city", ASCENDING), ("timestamp", ASCENDING)], False)],
    "daily_stats": [([("site_id", ASCENDING), ("date", ASCENDING)], True),
                    ([("date", ASCENDING)], False)],
    "monthly_stats": [([("site_id", ASCENDING), ("month", ASCENDING)], True)],
    "city_daily": [([("city", ASCENDING), ("date", ASCENDING)], True)],
    "hotspots": [([("site_id", ASCENDING)], True), ([("hotspot_rank", ASCENDING)], False)],
    "forecasts": [([("site_id", ASCENDING), ("date", ASCENDING)], True)],
    "sites": [([("site_id", ASCENDING)], True), ([("location", GEOSPHERE)], False)],
}


def get_client(timeout_ms: int = 3000) -> MongoClient:
    return MongoClient(load_config()["mongodb"]["uri"], serverSelectionTimeoutMS=timeout_ms)


def get_db(client: MongoClient | None = None):
    return (client or get_client())[load_config()["mongodb"]["database"]]


def is_available() -> bool:
    try:
        get_client(1500).admin.command("ping")
        return True
    except PyMongoError:
        return False


def _clean(value):
    """Make a value BSON-serialisable."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.to_pydatetime()
    if isinstance(value, date) and not isinstance(value, datetime):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, np.ndarray):
        return [_clean(v) for v in value.tolist()]
    return value


def records(df: pd.DataFrame) -> list[dict]:
    return [{k: _clean(v) for k, v in row.items()} for row in df.to_dict("records")]


def _write_partition(rows, uri: str, db_name: str) -> None:
    """Runs on Spark executors: bulk insert one partition."""
    from pymongo import MongoClient as _Client
    client = _Client(uri)
    coll = client[db_name]["hourly_measurements"]
    batch = []
    for row in rows:
        doc = row.asDict()
        doc["date"] = datetime(doc["date"].year, doc["date"].month, doc["date"].day)
        batch.append(doc)
        if len(batch) == 5000:
            coll.insert_many(batch, ordered=False)
            batch = []
    if batch:
        coll.insert_many(batch, ordered=False)
    client.close()


def _ensure_indexes(db) -> None:
    for coll, specs in INDEXES.items():
        for keys, unique in specs:
            db[coll].create_index(keys, unique=unique)


def load_all(spark) -> dict[str, int]:
    cfg = load_config()
    processed, raw = Path(cfg["paths"]["processed"]), Path(cfg["paths"]["raw"])
    client = get_client()
    client.admin.command("ping")
    db = get_db(client)
    counts = {}

    # sites with GeoJSON for geospatial queries ($near / $geoWithin)
    sites = json.loads((raw / "sites.json").read_text())
    db.sites.drop()
    db.sites.insert_many([{"site_id": s["id"], "name": s["name"], "city": s["city"],
                           "zone": s.get("zone"), "lat": s["lat"], "lon": s["lon"],
                           "location": {"type": "Point", "coordinates": [s["lon"], s["lat"]]}}
                          for s in sites])
    counts["sites"] = len(sites)

    # big table: distributed write from Spark executors
    db.hourly_measurements.drop()
    hourly = spark.read.parquet(str(processed / "hourly_clean"))
    uri, db_name = cfg["mongodb"]["uri"], cfg["mongodb"]["database"]
    hourly.repartition(8).foreachPartition(lambda rows: _write_partition(rows, uri, db_name))
    counts["hourly_measurements"] = db.hourly_measurements.estimated_document_count()

    for coll, name in DATASETS.items():
        path = processed / name
        if not path.exists():
            log.warning("Skipping %s: %s not found", coll, path)
            continue
        df = pd.read_parquet(path)
        db[coll].drop()
        if len(df):
            db[coll].insert_many(records(df))
        counts[coll] = len(df)

    _ensure_indexes(db)
    meta_path = raw / "ingestion_meta.json"
    db.pipeline_runs.insert_one({
        "finished_at": datetime.now(),
        "ingestion": json.loads(meta_path.read_text()) if meta_path.exists() else {},
        "counts": counts,
    })
    log.info("MongoDB load complete: %s", counts)
    return counts
