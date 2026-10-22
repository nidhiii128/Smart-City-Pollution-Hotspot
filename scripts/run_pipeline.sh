#!/usr/bin/env bash
# Start MongoDB (if needed) and run the full pipeline. Extra args are passed through,
# e.g.  ./scripts/run_pipeline.sh --skip-ingest   or   --source synthetic
set -euo pipefail
cd "$(dirname "$0")/.."
./scripts/start_mongo.sh || echo "Continuing without MongoDB (dashboard will use Parquet)"
.venv/bin/python -m src.pipeline "$@"
