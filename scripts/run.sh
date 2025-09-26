#!/usr/bin/env bash
# Helper script to create venv, install requirements, and run CLI or web UI.
# Usage:
#   bash scripts/run.sh --install --web
#   bash scripts/run.sh --install --cli -- input.wav output.txt
#   bash scripts/run.sh --web

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

INSTALL=0
WEB=0
CLI=0
INSTALL_FFMPEG=0

# parse args
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --install) INSTALL=1; shift ;;
    --web) WEB=1; shift ;;
    --cli) CLI=1; shift ;;
    --ffmpeg) INSTALL_FFMPEG=1; shift ;;
    --) shift; ARGS=("$@"); break ;;
    -h|--help)
      cat <<'USAGE'
Usage: scripts/run.sh [--install] [--web] [--cli] [--ffmpeg] -- [cli args]

--install: create venv and pip install -r requirements.txt
--web: start the Flask web UI (webapp.py)
--cli: run the CLI script (script.py) — any args after -- are passed to the script
--ffmpeg: on macOS, try to brew install ffmpeg when --install is used
USAGE
      exit 0
      ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ $INSTALL -eq 1 ]]; then
  echo "Creating virtualenv at $VENV_DIR (if missing)..."
  if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
  fi
  echo "Activating venv and installing requirements..."
  "$PIP" install --upgrade pip
  "$PIP" install -r "$ROOT_DIR/requirements.txt"

  if [[ $INSTALL_FFMPEG -eq 1 ]]; then
    if [[ "$(uname -s)" == "Darwin" ]]; then
      if command -v brew >/dev/null 2>&1; then
        echo "Installing ffmpeg via brew..."
        brew install ffmpeg
      else
        echo "Homebrew not found; please install FFmpeg manually (brew install ffmpeg)"
      fi
    else
      echo "Please install ffmpeg via your system package manager (apt, yum, pacman, etc.)"
    fi
  fi
fi

if [[ $WEB -eq 1 ]]; then
  echo "Starting web UI (http://localhost:5001)..."
  exec "$PYTHON" "$ROOT_DIR/webapp.py"
fi

if [[ $CLI -eq 1 ]]; then
  if [[ ${#ARGS[@]} -eq 0 ]]; then
    echo "No CLI arguments provided. Example: -- cli input.wav output.txt"
    exit 1
  fi
  echo "Running CLI: $PYTHON $ROOT_DIR/script.py ${ARGS[*]}"
  exec "$PYTHON" "$ROOT_DIR/script.py" "${ARGS[@]}"
fi

cat <<EOF
No action requested. Examples:
  bash scripts/run.sh --install --web
  bash scripts/run.sh --install --cli -- input.wav output.txt
EOF
exit 0
