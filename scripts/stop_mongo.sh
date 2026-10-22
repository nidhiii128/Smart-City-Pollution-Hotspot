#!/usr/bin/env bash
cd "$(dirname "$0")/.."
if command -v docker >/dev/null 2>&1 && docker compose ps -q mongo 2>/dev/null | grep -q .; then
  docker compose down; exit 0
fi
MONGOD="$(command -v mongod || ls ~/.local/mongodb/*/bin/mongod 2>/dev/null | head -1)"
"$MONGOD" --dbpath data/mongodb --shutdown
