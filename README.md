# Smart City Pollution Hotspot Detection Using Big Data Analytics

Finds air-pollution hotspots across **Mumbai and Thane** and forecasts air quality for the next 7 days.
Hourly air-quality and weather data for 34 monitoring sites goes through **Apache Spark** (cleaning,
CPCB AQI, K-Means clustering, Getis-Ord Gi\* hotspot statistics, Gradient-Boosted Tree forecasting).
The results are stored in **MongoDB** and shown on an interactive **Streamlit** dashboard with heatmaps.

```
 ┌───────────────┐   ┌──────────────────────────────────────────┐   ┌──────────┐   ┌────────────┐
 │ Data sources  │   │            Apache Spark (PySpark)         │   │ MongoDB  │   │ Dashboard  │
 │ • OpenAQ v3   │──▶│ ETL ─▶ AQI ─▶ Aggregates ─▶ Hotspots ─▶ ML │──▶│ 14 colls │──▶│ Streamlit  │
 │ • Open-Meteo  │   │ clean   CPCB   daily/monthly  K-Means   GBT │   │ indexed  │   │ + Plotly   │
 │ • simulator   │   │ dedupe  NAQI   diurnal/city   Gi*      7-day│   │ 2dsphere │   │ heatmaps   │
 └───────────────┘   └──────────────────────────────────────────┘   └──────────┘   └────────────┘
      raw zone (Parquet)          processed zone (Parquet)
```

## Quick start

Requirements: Python 3.10+ and **Java 17 or 21** (Spark 4). MongoDB is optional, but recommended.

```bash
cd pollution-hotspot-detection
./scripts/setup.sh                 # virtualenv + dependencies
./scripts/start_mongo.sh           # local mongod, or Docker (docker compose up -d)
./scripts/run_pipeline.sh          # ingest → Spark ETL → hotspots → forecast → MongoDB  (~5 min)
./scripts/run_dashboard.sh         # open http://localhost:8501
```

If MongoDB isn't running, the pipeline still finishes and the dashboard reads the Parquet outputs
directly. The sidebar shows which serving layer is in use.

## Data sources

Set `ingestion.source` in `config/config.yaml`, or pass `--source`:

| Source | What it is | Needs |
|---|---|---|
| `openaq` | **Real ground-station measurements** (CPCB/MPCB stations on OpenAQ v3). Stations within 25 km of Mumbai and Thane are found automatically. | Free API key from https://explore.openaq.org/register, set as `export OPENAQ_API_KEY=...` |
| `hybrid` *(default without a key)* | Real **CAMS** air-quality reanalysis and real **ERA5/forecast weather** from Open-Meteo, downscaled to each site with land-use emission factors | Internet |
| `openmeteo` | Raw CAMS grid. It is coarse: only about 3 distinct cells cover the whole area | Internet |
| `synthetic` | A physics-inspired simulator (seasons, monsoon washout, traffic peaks, sea breeze). Works offline | Nothing |

`auto` uses OpenAQ when a key is set, then hybrid, then synthetic.

> **Note on `hybrid`:** CAMS can't separate sites a few km apart. Hybrid mode keeps the real
> regional signal (seasonality, pollution episodes, weather response) and adds street-level
> variation using each site's land-use class (industrial / traffic / commercial / residential /
> coastal / green). It also halves CAMS ozone, because CAMS over-predicts ozone over South Asia.
> Site differences in this mode are therefore **modelled, not measured**. Use `openaq` for
> measured station-level results.

## Pipeline stages

| Stage | Module | What it does |
|---|---|---|
| Ingest | `src/ingestion/` | Downloads hourly PM2.5, PM10, NO₂, SO₂, CO, O₃ plus temperature, humidity, wind, rain and pressure. Writes to `data/raw/hourly/site_id=*/` as Parquet. `--incremental` fetches only new hours. |
| ETL | `src/processing/etl.py` | Spark: deduplicates, nulls invalid readings, caps each site at its 99.9th percentile, forward-fills gaps of up to 3 h, computes CPCB 24 h/8 h rolling averages, sub-indices, **AQI**, AQI category, dominant pollutant and season. Builds daily, monthly, diurnal, city and per-site summary tables. |
| Hotspots | `src/processing/hotspots.py` | **K-Means** (Spark MLlib) on standardised per-site particulate and AQI features (PM2.5, PM10, P95, exceedance rate, mean AQI). k is picked by silhouette score, and clusters are ranked into *Critical Hotspot → Clean Zone*. **Getis-Ord Gi\*** (8 km band) tests whether each hotspot is spatially significant. Also clusters each season separately and computes a combined 0–100 hotspot score. |
| Forecast | `src/processing/forecast.py` | A **GBTRegressor** (Spark MLlib, deliberately small: depth-2 trees) per target (PM2.5, AQI). It predicts the day-to-day change from lags, the 7-day mean, weather for the predicted day and day of week. It is validated on a chronological hold-out against a persistence baseline. The final model is rolled forward recursively for 7 days using Open-Meteo weather forecasts, with ~80% intervals. |
| Store | `src/storage/mongo_store.py` | Writes the hourly table in parallel from Spark executors (`foreachPartition` bulk inserts), plus the analytic tables. Adds compound indexes and a `2dsphere` index on sites. |
| Dashboard | `dashboard/` | Overview KPIs, hotspot heatmap and cluster map, ranking table, seasonal shift, time series, monthly heatmap, daily cycle, weather correlation, forecast map and site outlook, model metrics, and a raw-data explorer with CSV export. |

Run individual stages:

```bash
.venv/bin/python -m src.pipeline --skip-ingest            # reprocess existing raw data
.venv/bin/python -m src.pipeline --stages hotspots,mongo
.venv/bin/python -m src.pipeline --source synthetic       # fully offline
.venv/bin/python -m src.pipeline --incremental            # near real-time refresh
```

For near real-time updates, schedule `scripts/realtime_refresh.sh` hourly with cron (see the comment in the file).

## Results: real OpenAQ data, Oct 2025 – Oct 2026

* 44 CPCB/MPCB stations · 290,155 valid hourly records · 11,902 station-days
* **Hotspots** (K-Means k = 4 on particulate/AQI features, plus Getis-Ord Gi\*): Bandra Kurla Complex,
  Chakala-Andheri East, Deonar, Sion, Shivaji Nagar and Siddharth Nagar-Worli form the critical cluster.
  Andheri East, Vile Parle, Sion, Kurla and the airport also form a statistically significant spatial cluster
  (Gi\* 95%). The cleanest stations are Pimpleshwar Mandir (Thane), Vithalwadi, Kalu Nagar (Dombivli) and Bhandup West.
* Pollution varies less between stations than you might expect: annual mean PM2.5 ranges from about 14 to 43 µg/m³, and both
  cities average about 32 µg/m³. 13% of station-days exceed the 24 h PM2.5 standard (60 µg/m³).
* **Forecast:** the GBT model scores PM2.5 R² 0.80 (RMSE 6.6 µg/m³) and AQI R² 0.82 (RMSE 13.0). That is only
  0.8–1.8% better than persistence, so next-day air quality is mostly "same as today". The model
  configuration was chosen after looking at this hold-out, so treat the margin as roughly zero.

The earlier *hybrid* run (modelled data) gave GBT R² 0.83 and different hotspots (Kalyan, Ulhasnagar, Dombivli).
That data was downscaled with land-use factors, so use the OpenAQ results for any conclusions.

## MongoDB collections

`sites` (GeoJSON + 2dsphere), `hourly_measurements`, `daily_stats`, `monthly_stats`, `diurnal_profile`,
`city_daily`, `hotspots`, `cluster_profiles`, `seasonal_hotspots`, `forecasts`, `model_metrics`,
`feature_importance`, `test_predictions`, `pipeline_runs`. Database: `smart_city_pollution`
(override with the `MONGO_URI` env var).

Example queries (mongosh):

```js
use smart_city_pollution
db.hotspots.find({hotspot_label: "Critical Hotspot"}, {site_name: 1, mean_aqi: 1, gi_class: 1})
db.sites.find({location: {$near: {$geometry: {type: "Point", coordinates: [72.87, 19.07]}, $maxDistance: 5000}}})
db.hourly_measurements.aggregate([{$match: {aqi: {$gt: 300}}}, {$group: {_id: "$site_name", hours: {$sum: 1}}}, {$sort: {hours: -1}}])
```

## Project layout

```
config/config.yaml        sites, date range, model and Spark settings
src/ingestion/            openaq.py, openmeteo.py, synthetic.py, ingest.py
src/processing/           spark_session.py, etl.py, hotspots.py, forecast.py
src/storage/mongo_store.py
src/aqi.py                CPCB NAQI breakpoints and categories
src/pipeline.py           orchestrator (CLI)
dashboard/app.py, data.py Streamlit app and data-access layer (MongoDB → Parquet fallback)
scripts/                  setup, MongoDB start/stop, pipeline, dashboard, real-time refresh
docker-compose.yml        MongoDB 8 container
```

## Configuration

Edit `config/config.yaml` to change the date range (`start_date`; `end_date: auto` means yesterday),
add sites, or tune the clustering range, forecast horizon and Spark memory. Environment overrides:
`PHD_START_DATE`, `PHD_END_DATE`, `PHD_SOURCE`, `MONGO_URI`, `OPENAQ_API_KEY`.
To run on a cluster, set `spark.master` (for example `spark://host:7077` or `yarn`).

## Notes on real (OpenAQ) data

* **Download time.** OpenAQ's free tier allows about 60 requests a minute, and each station has several sensors.
  3 months for all 41 active stations takes about 1 hour, and a full year about 3 hours. Downloads are cached per
  station in `data/raw/cache/openaq/`, so an interrupted run resumes where it stopped. Use
  `PHD_START_DATE=YYYY-MM-DD` to choose the window.
* **Reporting lag.** CPCB stations reach OpenAQ a few days late. The forecast starts from the latest
  day most stations have reported.
* **Sensor faults.** CPCB feeds contain a lot of bad data. ETL removes:
  stuck sensors (the same value for 12 h or more, e.g. PM2.5 = 985 for weeks), readings at the monitor's full scale
  (PM2.5 ≥ 950, PM10 ≥ 1100 µg/m³), isolated spikes (more than 4× the ±12 h rolling median), and
  days with fewer than 16 valid hours (CPCB rule). Remaining outliers are capped per site. CO arrives in mg/m³ but
  is labelled µg/m³, so the ingestion corrects it.
* **Model selection.** The GBT forecast is used only if it beats persistence on the hold-out. Otherwise the
  pipeline forecasts with persistence, and the dashboard says which method was used. With 3 months of monsoon
  data GBT lost; with a full year it narrowly wins.
* **Java.** Spark 4 needs Java 17 or 21. If `JAVA_HOME` points to a newer JDK (for example Java 25 from SDKMAN),
  the pipeline switches to an installed Java 21/17 automatically.
