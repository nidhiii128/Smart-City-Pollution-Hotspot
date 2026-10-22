"""Data access for the dashboard: MongoDB first, Parquet files as a fallback."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.storage import mongo_store  # noqa: E402

CFG = load_config()
PROCESSED = Path(CFG["paths"]["processed"])
RAW = Path(CFG["paths"]["raw"])
DATE_COLS = {"date", "timestamp"}


@st.cache_resource(ttl=60)
def backend() -> str:
    return "MongoDB" if mongo_store.is_available() else "Parquet"


def _fix_dates(df: pd.DataFrame) -> pd.DataFrame:
    for c in DATE_COLS & set(df.columns):
        df[c] = pd.to_datetime(df[c])
    return df


@st.cache_data(ttl=300, show_spinner=False)
def load(collection: str) -> pd.DataFrame:
    if backend() == "MongoDB":
        db = mongo_store.get_db()
        df = pd.DataFrame(list(db[collection].find({}, {"_id": 0})))
    else:
        name = mongo_store.DATASETS.get(collection, collection)
        path = PROCESSED / name
        if collection == "sites":
            sites = json.loads((RAW / "sites.json").read_text())
            df = pd.DataFrame(sites).rename(columns={"id": "site_id"})
        elif path.exists():
            df = pd.read_parquet(path)
        else:
            df = pd.DataFrame()
    return _fix_dates(df)


@st.cache_data(ttl=300, show_spinner=False)
def hourly(site_ids: tuple[str, ...], start: datetime, end: datetime) -> pd.DataFrame:
    """Hourly measurements for a handful of sites - pushed down to the database."""
    cols = ["site_id", "site_name", "city", "timestamp", "pm2_5", "pm10", "no2", "so2", "co",
            "o3", "aqi", "aqi_category", "dominant_pollutant", "temperature", "humidity",
            "wind_speed", "precipitation"]
    if backend() == "MongoDB":
        cur = mongo_store.get_db().hourly_measurements.find(
            {"site_id": {"$in": list(site_ids)}, "timestamp": {"$gte": start, "$lte": end}},
            {"_id": 0, **{c: 1 for c in cols}}).sort("timestamp", 1)
        df = pd.DataFrame(list(cur))
    else:
        df = pd.read_parquet(PROCESSED / "hourly_clean", columns=cols,
                             filters=[("site_id", "in", list(site_ids))])
        df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
    return _fix_dates(df) if len(df) else pd.DataFrame(columns=cols)


@st.cache_data(ttl=300, show_spinner=False)
def latest_readings() -> pd.DataFrame:
    """Most recent hourly reading per site."""
    if backend() == "MongoDB":
        pipeline = [{"$sort": {"timestamp": -1}},
                    {"$group": {"_id": "$site_id", "doc": {"$first": "$$ROOT"}}},
                    {"$replaceRoot": {"newRoot": "$doc"}}, {"$project": {"_id": 0}}]
        df = pd.DataFrame(list(mongo_store.get_db().hourly_measurements.aggregate(
            pipeline, allowDiskUse=True)))
    else:
        df = pd.read_parquet(PROCESSED / "hourly_clean")
        df = df.sort_values("timestamp").groupby("site_id").tail(1)
    return _fix_dates(df)


@st.cache_data(ttl=300, show_spinner=False)
def run_info() -> dict:
    meta_path = RAW / "ingestion_meta.json"
    info = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if backend() == "MongoDB":
        last = mongo_store.get_db().pipeline_runs.find_one(sort=[("finished_at", -1)])
        if last:
            info["loaded_at"] = last["finished_at"].isoformat(timespec="seconds")
    return info
