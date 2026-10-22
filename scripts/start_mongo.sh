#!/usr/bin/env bash
# Start a local MongoDB for the project.
# Order of preference: already running -> Docker -> local mongod binary.
set -euo pipefail
cd "$(dirname "$0")/.."

if (exec 3<>/dev/tcp/127.0.0.1/27017) 2>/dev/null; then
  echo "MongoDB already listening on localhost:27017"; exit 0
fi

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  docker compose up -d mongo && echo "MongoDB started in Docker"; exit 0
fi

MONGOD="$(command -v mongod || ls ~/.local/mongodb/*/bin/mongod 2>/dev/null | head -1 || true)"
if [[ -z "$MONGOD" ]]; then
  echo "mongod not found. Install MongoDB or Docker (see README)." >&2; exit 1
fi
mkdir -p data/mongodb
"$MONGOD" --dbpath data/mongodb --bind_ip 127.0.0.1 --port 27017 \
  --fork --logpath data/mongodb/mongod.log
echo "MongoDB started from $MONGOD (data in data/mongodb)"
