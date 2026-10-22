"""OpenAQ v3 client for ground-station measurements.

Requires a free API key from https://explore.openaq.org/register, exported as
OPENAQ_API_KEY. Stations within `radius_m` of each city centre that are still
reporting during the requested window are discovered automatically and their
hourly-averaged sensor data is downloaded.

The download is resumable: each finished station is cached under
data/raw/cache/openaq/<start>_<end>/, so re-running after an interruption
only fetches the stations that are missing. Transient server errors are
retried; a month of data that keeps failing is skipped with a warning
instead of aborting the whole run.
"""
from __future__ import annotations

import logging
import math
import re
import time
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

BASE = "https://api.openaq.org/v3"
CITY_CENTRES = {"Mumbai": (19.0760, 72.8777), "Thane": (19.2183, 72.9781)}
PARAM_MAP = {"pm25": "pm2_5", "pm10": "pm10", "no2": "no2", "so2": "so2", "co": "co", "o3": "o3"}
# molecular weights for ppm/ppb -> ug/m3 at 25 degC, 1 atm (molar volume 24.45 L)
MOL_WEIGHT = {"no2": 46.01, "so2": 64.07, "co": 28.01, "o3": 48.00}
MAX_ATTEMPTS = 5


class OpenAQError(RuntimeError):
    """A request that still failed after all retries."""


class OpenAQClient:
    def __init__(self, api_key: str, timeout: int = 60):
        if not api_key:
            raise ValueError("OPENAQ_API_KEY is not set")
        self.session = requests.Session()
        self.session.headers["X-API-Key"] = api_key
        self.timeout = timeout

    def _get(self, path: str, params: dict) -> dict:
        last_error = ""
        for attempt in range(MAX_ATTEMPTS):
            try:
                r = self.session.get(f"{BASE}{path}", params=params, timeout=self.timeout)
            except requests.RequestException as exc:  # timeouts, dropped connections
                last_error = str(exc)
            else:
                if r.status_code in (401, 403):
                    raise RuntimeError(
                        f"OpenAQ rejected the API key (HTTP {r.status_code}). Set OPENAQ_API_KEY to "
                        "the key shown at https://explore.openaq.org/account after registering.")
                if r.status_code == 429:  # 60 req/min on the free tier
                    wait = int(r.headers.get("x-ratelimit-reset", 10 * (attempt + 1)))
                    log.info("OpenAQ rate limit hit, sleeping %ss", wait)
                    time.sleep(wait)
                    continue
                if r.ok:
                    time.sleep(1.05)
                    return r.json()
                if r.status_code < 500 and r.status_code != 408:
                    r.raise_for_status()  # a genuine client error - retrying won't help
                last_error = f"HTTP {r.status_code}"
            wait = 5 * 2 ** attempt
            log.info("OpenAQ %s failed (%s), retry %d/%d in %ss",
                     path, last_error, attempt + 1, MAX_ATTEMPTS - 1, wait)
            time.sleep(wait)
        raise OpenAQError(f"{path} failed after {MAX_ATTEMPTS} attempts ({last_error})")

    def locations(self, radius_m: int, start: str) -> list[dict]:
        sites: dict[int, dict] = {}
        inactive = 0
        for lat, lon in CITY_CENTRES.values():
            data = self._get("/locations", {"coordinates": f"{lat},{lon}",
                                            "radius": radius_m, "limit": 1000})
            for loc in data.get("results", []):
                if loc["id"] in sites or loc.get("isMobile"):
                    continue
                last = (loc.get("datetimeLast") or {}).get("utc")
                if not last or last[:10] < start:  # stopped reporting before our window
                    inactive += 1
                    continue
                sensors = {s["id"]: s["parameter"] for s in loc.get("sensors", [])
                           if s["parameter"]["name"] in PARAM_MAP}
                if not sensors:
                    continue
                s_lat, s_lon = loc["coordinates"]["latitude"], loc["coordinates"]["longitude"]
                name = loc.get("name") or f"OpenAQ {loc['id']}"
                sites[loc["id"]] = {
                    "id": f"OAQ{loc['id']}",
                    "name": re.sub(r"\s*-\s*(MPCB|IITM|CPCB)\b", "", name).strip(),
                    "city": "",  # set in assign_cities_and_unique_names
                    "lat": s_lat,
                    "lon": s_lon,
                    "zone": "station",
                    "sensors": sensors,
                }
        log.info("OpenAQ: %d active stations with pollutant sensors (%d inactive skipped)",
                 len(sites), inactive)
        return assign_cities_and_unique_names(list(sites.values()))

    def sensor_hours(self, sensor_id: int, start: str, end: str) -> pd.DataFrame:
        """Hourly values, requested one month at a time to keep each request small."""
        rows = []
        months = pd.date_range(pd.Timestamp(start).replace(day=1), end, freq="MS")
        for m_start in months:
            w_from = max(m_start, pd.Timestamp(start))
            w_to = min(m_start + pd.offsets.MonthEnd(1), pd.Timestamp(end))
            page = 1
            while True:
                try:
                    data = self._get(f"/sensors/{sensor_id}/hours", {
                        "datetime_from": f"{w_from:%Y-%m-%d}T00:00:00Z",
                        "datetime_to": f"{w_to:%Y-%m-%d}T23:59:59Z",
                        "limit": 1000, "page": page})
                except OpenAQError as exc:
                    log.warning("Skipping sensor %s for %s: %s", sensor_id, f"{m_start:%Y-%m}", exc)
                    break
                results = data.get("results", [])
                rows.extend((r["period"]["datetimeFrom"]["utc"], r["value"]) for r in results)
                if len(results) < 1000:
                    break
                page += 1
        df = pd.DataFrame(rows, columns=["timestamp", "value"])
        if not df.empty:
            df["timestamp"] = (pd.to_datetime(df["timestamp"], utc=True)
                               .dt.tz_convert("Asia/Kolkata").dt.tz_localize(None))
        return df


# Thane district & Navi Mumbai localities; anything named "..., Mumbai" is Mumbai
THANE_KEYWORDS = ("navi mumbai", "thane", "kalyan", "dombivli", "ulhasnagar", "bhiwandi",
                  "ambernath", "badlapur", "mira", "bhayandar", "belapur", "vashi", "airoli",
                  "taloja", "kalamboli", "mahape", "sanpada", "nerul", "kharghar")


def _city_for(name: str, lat: float, lon: float) -> str:
    lower = name.lower()
    if any(k in lower for k in THANE_KEYWORDS):
        return "Thane"
    if "mumbai" in lower:
        return "Mumbai"
    return min(CITY_CENTRES, key=lambda c: math.dist((lat, lon), CITY_CENTRES[c]))


def assign_cities_and_unique_names(stations: list[dict]) -> list[dict]:
    """Set each station's city from its name and make duplicate names unique."""
    counts: dict[str, int] = {}
    for st in stations:
        st["city"] = _city_for(st["name"], st["lat"], st["lon"])
        counts[st["name"]] = counts.get(st["name"], 0) + 1
    for st in stations:
        if counts[st["name"]] > 1:
            st["name"] = f"{st['name']} (#{st['id'].removeprefix('OAQ')})"
    return stations


def _to_ugm3(param: str, units: str, values: pd.Series) -> pd.Series:
    units = units.lower()
    if param in MOL_WEIGHT and units == "ppm":
        return values * 1000 * MOL_WEIGHT[param] / 24.45
    if param in MOL_WEIGHT and units == "ppb":
        return values * MOL_WEIGHT[param] / 24.45
    if param == "co" and units in ("mg/m³", "mg/m3"):
        return values * 1000
    return values


def fetch(start: str, end: str, radius_m: int, api_key: str, timeout: int = 60,
          cache_dir: Path | None = None) -> tuple[pd.DataFrame, list[dict]]:
    client = OpenAQClient(api_key, timeout)
    stations = client.locations(radius_m, start)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for i, st in enumerate(stations, 1):
        cached = cache_dir / f"{st['id']}.parquet" if cache_dir is not None else None
        if cached is not None and cached.exists():
            merged = pd.read_parquet(cached)
            log.info("[%d/%d] %s: %d hourly rows (cached)", i, len(stations), st["name"], len(merged))
        else:
            t0 = time.time()
            per_param = []
            for sensor_id, param in st["sensors"].items():
                col = PARAM_MAP[param["name"]]
                df = client.sensor_hours(sensor_id, start, end)
                if df.empty:
                    continue
                df[col] = _to_ugm3(col, param.get("units", ""), df.pop("value"))
                if col == "co" and df[col].median() < 20:
                    # CPCB reports CO in mg/m3 but OpenAQ labels it ug/m3 (ambient CO is ~500-3000 ug/m3)
                    df[col] *= 1000
                per_param.append(df.groupby("timestamp")[col].mean())
            # several sensors can report the same pollutant - average them
            merged = (pd.concat(per_param, axis=1).T.groupby(level=0).mean().T.reset_index()
                      .rename(columns={"index": "timestamp"})) if per_param else pd.DataFrame()
            if cached is not None:
                merged.to_parquet(cached, index=False, coerce_timestamps="us")
            log.info("[%d/%d] %s: %d hourly rows (%.0fs)", i, len(stations), st["name"],
                     len(merged), time.time() - t0)
        if not merged.empty:
            merged["site_id"] = st["id"]
            frames.append(merged)

    for st in stations:
        st.pop("sensors", None)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), stations
