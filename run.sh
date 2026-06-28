#!/usr/bin/env bash
# Docaware - one-click launcher (Linux/macOS, e.g. the ADTC Ubuntu target).
# Requires: models downloaded (bash download_model.sh) and deps installed
# (pip install -r app/requirements.txt).
set -e
cd "$(dirname "$0")/app"

echo "Starting Docaware on http://127.0.0.1:8000  (Ctrl+C to stop)"
python -m docaware serve --port 8000 &
SERVER_PID=$!
sleep 5
( xdg-open http://127.0.0.1:8000 2>/dev/null || open http://127.0.0.1:8000 2>/dev/null || true )
wait $SERVER_PID
