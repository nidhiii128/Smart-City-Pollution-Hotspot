"""Indian National Air Quality Index (CPCB NAQI) helpers.

Breakpoints follow the CPCB 2014 NAQI specification. Concentrations are in
ug/m3 except CO which is mg/m3. PM / NO2 / SO2 use 24-hour averages and
CO / O3 use 8-hour averages.
"""
from __future__ import annotations

import math

# (conc_low, conc_high, aqi_low, aqi_high). Interpolation uses the previous
# bracket's upper bound as the lower bound so the index is continuous.
BREAKPOINTS: dict[str, list[tuple[float, float, int, int]]] = {
    "pm2_5": [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200),
              (91, 120, 201, 300), (121, 250, 301, 400), (251, 500, 401, 500)],
    "pm10": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200),
             (251, 350, 201, 300), (351, 430, 301, 400), (431, 600, 401, 500)],
    "no2": [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200),
            (181, 280, 201, 300), (281, 400, 301, 400), (401, 800, 401, 500)],
    "so2": [(0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200),
            (381, 800, 201, 300), (801, 1600, 301, 400), (1601, 2400, 401, 500)],
    "co": [(0, 1.0, 0, 50), (1.1, 2.0, 51, 100), (2.1, 10, 101, 200),
           (10.1, 17, 201, 300), (17.1, 34, 301, 400), (34.1, 50, 401, 500)],
    "o3": [(0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200),
           (169, 208, 201, 300), (209, 748, 301, 400), (749, 1000, 401, 500)],
}

CATEGORIES = [
    (50, "Good", "#2e9d4e"),
    (100, "Satisfactory", "#9acd32"),
    (200, "Moderate", "#f2c40f"),
    (300, "Poor", "#f08c00"),
    (400, "Very Poor", "#d9480f"),
    (10_000, "Severe", "#7b1f1f"),
]

# Daily safe limits (NAAQS India, 24h) used for exceedance statistics.
NAAQS_24H = {"pm2_5": 60.0, "pm10": 100.0, "no2": 80.0, "so2": 80.0, "co": 2.0, "o3": 100.0}


def sub_index(pollutant: str, conc: float | None) -> float | None:
    if conc is None or (isinstance(conc, float) and math.isnan(conc)) or conc < 0:
        return None
    prev_hi = 0.0
    for _, hi, ilo, ihi in BREAKPOINTS[pollutant]:
        if conc <= hi:
            return ilo + (ihi - ilo) * (conc - prev_hi) / (hi - prev_hi)
        prev_hi = hi
    return 500.0


def category(aqi: float | None) -> str:
    if aqi is None or (isinstance(aqi, float) and math.isnan(aqi)):
        return "Unknown"
    for upper, name, _ in CATEGORIES:
        if aqi <= upper:
            return name
    return "Severe"


def category_color(name: str) -> str:
    return next((c for _, n, c in CATEGORIES if n == name), "#888888")
