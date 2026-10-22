"""Offline simulator producing realistic hourly air-quality + weather data.

Used when there is no internet access. The model reproduces the main drivers
of Mumbai/Thane air quality: winter inversions and calm winds (Nov-Feb),
monsoon wash-out (Jun-Sep), morning/evening traffic peaks, sea-breeze
dilution at coastal sites, and higher baselines at industrial clusters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# multiplicative emission factor per pollutant for each land-use zone
ZONE_PROFILE = {
    #               pm2_5 pm10  no2   so2   co    o3
    "industrial":  (1.45, 1.50, 1.30, 1.90, 1.30, 0.90),
    "traffic":     (1.30, 1.35, 1.60, 1.00, 1.55, 0.85),
    "commercial":  (1.10, 1.15, 1.30, 0.90, 1.20, 0.95),
    "residential": (1.00, 1.00, 1.00, 0.85, 1.00, 1.00),
    "coastal":     (0.75, 0.85, 0.80, 0.80, 0.80, 1.10),
    "green":       (0.70, 0.75, 0.65, 0.70, 0.75, 1.20),
}
BASE = {"pm2_5": 38.0, "pm10": 85.0, "no2": 32.0, "so2": 12.0, "co": 900.0, "o3": 45.0}


def _weather(idx: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    doy = idx.dayofyear.to_numpy()
    hour = idx.hour.to_numpy()
    month = idx.month.to_numpy()
    monsoon = np.isin(month, [6, 7, 8, 9])
    temp = 28 + 3 * np.sin(2 * np.pi * (doy - 80) / 365) + 3.5 * np.sin(2 * np.pi * (hour - 9) / 24)
    temp += rng.normal(0, 0.8, len(idx))
    humidity = np.clip(np.where(monsoon, 86, 62) - 12 * np.sin(2 * np.pi * (hour - 9) / 24)
                       + rng.normal(0, 5, len(idx)), 20, 100)
    wind = np.clip(np.where(monsoon, 18, 9) + 5 * np.sin(2 * np.pi * (hour - 11) / 24)
                   + rng.gamma(2, 1.5, len(idx)) - 3, 0.5, 45)
    wind_dir = np.where(hour.astype(int) % 24 < 11, 60, 270) + rng.normal(0, 30, len(idx))
    rain_p = np.where(monsoon, 0.35, 0.01)
    precip = np.where(rng.random(len(idx)) < rain_p, rng.gamma(1.4, 3.0, len(idx)), 0.0)
    pressure = 1008 - 4 * monsoon + rng.normal(0, 1.5, len(idx))
    return pd.DataFrame({"temperature": temp, "humidity": humidity, "wind_speed": wind,
                         "wind_direction": np.mod(wind_dir, 360), "precipitation": precip,
                         "pressure": pressure}, index=idx)


def generate(sites: list[dict], start: str, end: str, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h")
    city_weather = _weather(idx, rng)
    month = idx.month.to_numpy()
    hour = idx.hour.to_numpy()
    doy = idx.dayofyear.to_numpy()

    # seasonal factor: high in winter, very low in monsoon
    season = 1.0 + 0.75 * np.cos(2 * np.pi * (doy - 15) / 365)
    season = np.where(np.isin(month, [6, 7, 8, 9]), season * 0.55, season)
    # traffic double peak (09:00, 20:00) + night inversion trap
    diurnal = (1 + 0.35 * np.exp(-((hour - 9) ** 2) / 4) + 0.45 * np.exp(-((hour - 20) ** 2) / 5)
               + 0.15 * ((hour < 6) | (hour > 22)))
    weekday = np.where(idx.dayofweek.to_numpy() >= 5, 0.88, 1.0)
    # synoptic episodes (multi-day stagnation events) shared across the region
    episodes = np.exp(np.convolve(rng.normal(0, 0.35, len(idx)), np.ones(72) / 72 * 8, "same") * 0.5)

    frames = []
    for site in sites:
        w = city_weather.copy()
        w["temperature"] += rng.normal(0, 0.4) - (0.6 if site["zone"] == "green" else 0)
        w["wind_speed"] *= 1.25 if site["zone"] == "coastal" else 1.0
        dispersion = 10.0 / (w["wind_speed"].to_numpy() + 6.0)
        washout = np.exp(-0.18 * w["precipitation"].to_numpy())
        inland = 1 + 3.0 * max(0.0, site["lon"] - 72.82)  # sea breeze weaker inland
        site_noise = rng.normal(1.0, 0.06)
        prof = ZONE_PROFILE.get(site["zone"], ZONE_PROFILE["residential"])
        df = pd.DataFrame({"timestamp": idx, "site_id": site["id"]})
        for (pol, base), factor in zip(BASE.items(), prof):
            if pol == "o3":  # photochemical: peaks mid-afternoon, lower in monsoon
                shape = 1 + 0.8 * np.exp(-((hour - 14) ** 2) / 8)
                val = base * factor * shape * np.where(np.isin(month, [6, 7, 8, 9]), 0.6, 1.0)
            else:
                val = (base * factor * season * diurnal * weekday * dispersion * washout
                       * episodes * inland * site_noise)
            val *= rng.lognormal(0, 0.18, len(idx))
            df[pol] = np.round(val, 2)
        frames.append(pd.concat([df.reset_index(drop=True),
                                 w.reset_index(drop=True).round(2)], axis=1))

    out = pd.concat(frames, ignore_index=True)
    # sprinkle sensor gaps and spikes so the cleaning stage has work to do
    mask = rng.random(len(out)) < 0.015
    out.loc[mask, "pm2_5"] = np.nan
    spikes = rng.random(len(out)) < 0.0007
    out.loc[spikes, "pm10"] = -999.0
    return out


# CAMS over-predicts surface ozone over South Asia by roughly a factor of two
# compared with CPCB ground stations; applied only in hybrid mode.
CAMS_BIAS_CORRECTION = {"o3": 0.5}


def downscale(df: pd.DataFrame, sites: list[dict], seed: int = 42) -> pd.DataFrame:
    """Add local (sub-grid) variation to coarse CAMS model output.

    CAMS resolves the Mumbai-Thane region with only a handful of grid cells, so
    every site inside one cell receives an identical series. This applies the
    land-use emission factors above, a local traffic diurnal cycle and
    temporally correlated sensor-scale noise so that street-level hotspots
    (industrial belts, traffic junctions) become distinguishable. The regional
    signal (seasonality, episodes, weather response) stays the real CAMS one.
    """
    rng = np.random.default_rng(seed)
    site_map = {s["id"]: s for s in sites}
    df = df.copy()
    for pol, factor in CAMS_BIAS_CORRECTION.items():
        df[pol] = df[pol] * factor
    out = []
    for site_id, part in df.groupby("site_id", sort=False):
        part = part.sort_values("timestamp").copy()
        site = site_map[site_id]
        prof = ZONE_PROFILE.get(site["zone"], ZONE_PROFILE["residential"])
        hour = part["timestamp"].dt.hour.to_numpy()
        traffic = 1 + (0.25 if site["zone"] in ("traffic", "commercial") else 0.08) * (
            np.exp(-((hour - 9) ** 2) / 4) + np.exp(-((hour - 20) ** 2) / 5))
        wind = part["wind_speed"].fillna(part["wind_speed"].median()).to_numpy()
        local_disp = 1 + 0.25 * (8.0 / (wind + 4.0) - 0.6)  # stagnant air amplifies local sources
        inland = 1 + 1.2 * max(0.0, site["lon"] - 72.82)
        n = len(part)
        # AR(1) noise -> realistic, autocorrelated deviations from the grid mean
        eps = rng.normal(0, 0.06, n)
        ar = np.empty(n)
        ar[0] = eps[0]
        for i in range(1, n):
            ar[i] = 0.9 * ar[i - 1] + eps[i]
        site_bias = rng.normal(1.0, 0.05)
        for pol, factor in zip(BASE, prof):
            if pol not in part:
                continue
            local = factor * site_bias * np.exp(ar)
            if pol != "o3":
                local = local * traffic * local_disp * inland
            part[pol] = (part[pol].to_numpy() * local).round(2)
        out.append(part)
    return pd.concat(out, ignore_index=True)
