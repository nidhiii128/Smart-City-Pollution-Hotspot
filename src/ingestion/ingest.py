"""Stage 1 - data ingestion.

Collects hourly air-quality and weather data for every monitoring site and
lands it in the raw zone (data/raw) as Parquet, one file per site:

    data/raw/hourly/site_id=<ID>/part.parquet
    data/raw/sites.json
    data/raw/weather_forecast.parquet   (drives the 7-day prediction)
    data/raw/ingestion_meta.json

Usage:
    python -m src.ingestion.ingest                 # full historical load
    python -m src.ingestion.ingest --incremental   # only fetch new hours (near real-time)
    python -m src.ingestion.ingest --source synthetic

Sources: openaq (real ground stations, needs OPENAQ_API_KEY), hybrid (real CAMS
background + real weather, downscaled to street level by land-use), openmeteo
(raw CAMS grid, coarse) and synthetic (fully offline simulator).
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from src.config import load_config
from src.ingestion import openaq, openmeteo, synthetic

log = logging.getLogger("ingest")
COLUMNS = ["site_id", "timestamp", "pm2_5", "pm10", "no2", "so2", "co", "o3",
           "temperature", "humidity", "wind_speed", "wind_direction", "precipitation", "pressure"]


def _resolve_source(requested: str, cfg: dict) -> str:
    if requested != "auto":
        return requested
    if cfg["openaq_api_key"]:
        return "openaq"
    if openmeteo.is_reachable():
        return "hybrid"
    log.warning("No internet access - falling back to the synthetic simulator")
    return "synthetic"


def _existing_max_ts(hourly_dir: Path) -> pd.Timestamp | None:
    files = list(hourly_dir.glob("site_id=*/part.parquet"))
    if not files:
        return None
    return max(pd.read_parquet(f, columns=["timestamp"])["timestamp"].max() for f in files)


def _write_raw(df: pd.DataFrame, hourly_dir: Path, merge: bool) -> int:
    total = 0
    for site_id, part in df.groupby("site_id"):
        out = hourly_dir / f"site_id={site_id}" / "part.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        part = part.drop(columns="site_id")
        if merge and out.exists():
            part = pd.concat([pd.read_parquet(out), part])
        part = (part.drop_duplicates("timestamp", keep="last")
                    .sort_values("timestamp").reset_index(drop=True))
        part.to_parquet(out, index=False, coerce_timestamps="us")  # Spark can't read ns
        total += len(part)
    return total


def run(source: str | None = None, incremental: bool = False) -> dict:
    cfg = load_config()
    ing = cfg["ingestion"]
    raw = Path(cfg["paths"]["raw"])
    hourly_dir = raw / "hourly"
    timeout = ing["request_timeout_s"]
    start, end = ing["start_date"], ing["end_date"]

    meta_path = raw / "ingestion_meta.json"
    prev_meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    source = _resolve_source(source or ing["source"], cfg)

    if incremental:
        last = _existing_max_ts(hourly_dir)
        if last is not None and prev_meta.get("source") == source:
            start = (last.normalize()).date().isoformat()
            end = (date.today() - timedelta(days=1)).isoformat()
            log.info("Incremental load %s -> %s", start, end)
        else:
            incremental = False
    sites = cfg["sites"]
    log.info("Ingesting %s -> %s using source=%s", start, end, source)

    if source == "synthetic":
        df = synthetic.generate(sites, start, end, seed=cfg["clustering"]["seed"])
    elif source in ("openmeteo", "hybrid"):
        cache = raw / "cache" / f"openmeteo_{start}_{end}_{len(sites)}.parquet"
        if cache.exists():
            log.info("Using cached Open-Meteo download %s", cache.name)
            df = pd.read_parquet(cache)
        else:
            aq = openmeteo.fetch_air_quality(sites, start, end, timeout)
            wx = openmeteo.fetch_weather(sites, start, end, timeout)
            df = aq.merge(wx, on=["site_id", "timestamp"], how="left")
            cache.parent.mkdir(exist_ok=True)
            df.to_parquet(cache, index=False)
        if source == "hybrid":
            df = synthetic.downscale(df, sites, seed=cfg["clustering"]["seed"])
    elif source == "openaq":
        aq, sites = openaq.fetch(start, end, ing["openaq_radius_m"], cfg["openaq_api_key"], timeout,
                                 cache_dir=raw / "cache" / "openaq" / f"{start}_{end}")
        if aq.empty:
            raise RuntimeError("OpenAQ returned no measurements for Mumbai/Thane in this window")
        sites = [s for s in sites if s["id"] in set(aq["site_id"])]
        wx = openmeteo.fetch_weather(sites, start, end, timeout)
        df = aq.merge(wx, on=["site_id", "timestamp"], how="left")
    else:
        raise ValueError(f"Unknown source {source!r}")

    for col in COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[COLUMNS]
    if df.empty:
        raise RuntimeError(f"Source {source!r} returned no data; existing raw data left untouched")
    # replace the previous full load only now that the new download succeeded
    if not incremental and hourly_dir.exists():
        for f in hourly_dir.glob("site_id=*/part.parquet"):
            f.unlink()
    rows = _write_raw(df, hourly_dir, merge=incremental)

    (raw / "sites.json").write_text(json.dumps(sites, indent=2))

    try:
        if source == "synthetic":
            raise RuntimeError("offline")
        fc = openmeteo.fetch_weather_forecast(sites, cfg["prediction"]["horizon_days"], timeout)
        fc.to_parquet(raw / "weather_forecast.parquet", index=False, coerce_timestamps="us")
    except Exception as exc:  # noqa: BLE001 - forecast weather is optional
        log.info("Weather forecast unavailable (%s); prediction will use climatology", exc)
        (raw / "weather_forecast.parquet").unlink(missing_ok=True)

    meta = {"source": source, "start_date": ing["start_date"] if incremental else start,
            "end_date": end, "sites": len(sites), "rows": rows,
            "ingested_at": datetime.now().isoformat(timespec="seconds")}
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("Raw zone now holds %d hourly rows for %d sites", rows, len(sites))
    return meta


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["auto", "openaq", "hybrid", "openmeteo", "synthetic"])
    ap.add_argument("--incremental", action="store_true", help="append only new hours")
    args = ap.parse_args()
    print(json.dumps(run(args.source, args.incremental), indent=2))


if __name__ == "__main__":
    main()
