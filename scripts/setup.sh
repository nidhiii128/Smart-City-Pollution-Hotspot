#!/usr/bin/env bash
# One-time setup: virtualenv + Python dependencies.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
echo "Done. Java 17 or 21 is required by Spark: $(java -version 2>&1 | head -1)"
