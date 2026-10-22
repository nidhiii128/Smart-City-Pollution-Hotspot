"""Open-Meteo clients: CAMS air-quality reanalysis/forecast and ERA5 weather.

No API key is needed. Several coordinates are requested per call to keep the
number of HTTP requests small.
"""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta

import pandas as pd
import requests

log = logging.getLogger(__name__)

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

AQ_VARS = {
    "pm2_5": "pm2_5",
    "pm10": "pm10",
    "nitrogen_dioxide": "no2",
    "sulphur_dioxide": "so2",
    "carbon_monoxide": "co",
    "ozone": "o3",
}
WEATHER_VARS = {
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_direction",
    "precipitation": "precipitation",
    "surface_pressure": "pressure",
}
BATCH = 10


def _get(url: str, params: dict, timeout: int, retries: int = 4) -> list[dict]:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            r.raise_for_status()
            data = r.json()
            return data if isinstance(data, list) else [data]
        except requests.RequestException as exc:
            log.warning("Open-Meteo request failed (%s), retry %d", exc, attempt + 1)
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Open-Meteo request to {url} failed after {retries} retries")


def _fetch(url: str, sites: list[dict], variables: dict[str, str], extra: dict,
           timeout: int) -> pd.DataFrame:
    frames = []
    for i in range(0, len(sites), BATCH):
        chunk = sites[i:i + BATCH]
        params = {
            "latitude": ",".join(f"{s['lat']:.4f}" for s in chunk),
            "longitude": ",".join(f"{s['lon']:.4f}" for s in chunk),
            "hourly": ",".join(variables),
            "timezone": "Asia/Kolkata",
            **extra,
        }
        for site, payload in zip(chunk, _get(url, params, timeout)):
            hourly = payload["hourly"]
            df = pd.DataFrame({"timestamp": pd.to_datetime(hourly["time"])})
            for api_name, col in variables.items():
                df[col] = pd.to_numeric(pd.Series(hourly.get(api_name)), errors="coerce")
            df["site_id"] = site["id"]
            frames.append(df)
        time.sleep(0.5)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_air_quality(sites: list[dict], start: str, end: str, timeout: int = 60) -> pd.DataFrame:
    """Hourly pollutant concentrations (ug/m3) from the CAMS model via Open-Meteo."""
    log.info("Open-Meteo air quality %s -> %s for %d sites", start, end, len(sites))
    return _fetch(AQ_URL, sites, AQ_VARS, {"start_date": start, "end_date": end}, timeout)


def fetch_weather(sites: list[dict], start: str, end: str, timeout: int = 60) -> pd.DataFrame:
    """Hourly weather. ERA5 archive for older days + forecast API for the last week."""
    log.info("Open-Meteo weather %s -> %s for %d sites", start, end, len(sites))
    start_d, end_d = date.fromisoformat(start), date.fromisoformat(end)
    archive_end = min(end_d, date.today() - timedelta(days=6))
    frames = []
    if start_d <= archive_end:
        frames.append(_fetch(ARCHIVE_URL, sites, WEATHER_VARS,
                             {"start_date": start, "end_date": archive_end.isoformat()}, timeout))
    if end_d > archive_end:
        recent_start = max(start_d, archive_end + timedelta(days=1))
        frames.append(_fetch(FORECAST_URL, sites, WEATHER_VARS,
                             {"start_date": recent_start.isoformat(), "end_date": end}, timeout))
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(["site_id", "timestamp"], keep="last")


def fetch_weather_forecast(sites: list[dict], days: int = 7, timeout: int = 60) -> pd.DataFrame:
    """Hourly weather forecast for the prediction horizon."""
    log.info("Open-Meteo weather forecast for next %d days", days)
    return _fetch(FORECAST_URL, sites, WEATHER_VARS, {"forecast_days": days + 1}, timeout)


def is_reachable(timeout: int = 20, attempts: int = 3) -> bool:
    for _ in range(attempts):
        try:
            r = requests.get(AQ_URL, params={"latitude": 19.07, "longitude": 72.87,
                                             "hourly": "pm2_5", "forecast_days": 1}, timeout=timeout)
            if r.ok:
                return True
        except requests.RequestException as exc:
            log.info("Open-Meteo connectivity check failed: %s", exc)
        time.sleep(2)
    return False
