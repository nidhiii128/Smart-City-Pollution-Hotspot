#!/usr/bin/env bash
# Near real-time refresh: fetch new hours, recompute analytics, reload MongoDB.
# Schedule hourly with cron:
#   0 * * * * /path/to/pollution-hotspot-detection/scripts/realtime_refresh.sh >> /tmp/pollution_refresh.log 2>&1
set -euo pipefail
cd "$(dirname "$0")/.."
.venv/bin/python -m src.pipeline --incremental
