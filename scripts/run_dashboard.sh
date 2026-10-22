#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
./scripts/start_mongo.sh || true
exec .venv/bin/streamlit run dashboard/app.py "$@"
