#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

UVICORN_BIN=${UVICORN_BIN:-uvicorn}

exec "$UVICORN_BIN" streaming.server:streaming_app --host 0.0.0.0 --port "${STREAMING_PORT:-8001}" --reload
