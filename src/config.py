"""Configuration loader shared by every pipeline stage and the dashboard."""
from __future__ import annotations

import os
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(os.environ.get("PHD_CONFIG", ROOT / "config" / "config.yaml"))


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    ing = cfg["ingestion"]
    if ing.get("end_date", "auto") == "auto":
        ing["end_date"] = (date.today() - timedelta(days=1)).isoformat()
    if os.environ.get("PHD_START_DATE"):
        ing["start_date"] = os.environ["PHD_START_DATE"]
    if os.environ.get("PHD_END_DATE"):
        ing["end_date"] = os.environ["PHD_END_DATE"]
    if os.environ.get("PHD_SOURCE"):
        ing["source"] = os.environ["PHD_SOURCE"]

    cfg["mongodb"]["uri"] = os.environ.get("MONGO_URI", cfg["mongodb"]["uri"])
    cfg["openaq_api_key"] = os.environ.get("OPENAQ_API_KEY", "")

    for key, rel in cfg["paths"].items():
        p = (ROOT / rel).resolve()
        p.mkdir(parents=True, exist_ok=True)
        cfg["paths"][key] = str(p)
    return cfg


def sites_by_id() -> dict[str, dict]:
    return {s["id"]: s for s in load_config()["sites"]}
