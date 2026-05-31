#!/bin/bash
# Run server
# Usage: ./scripts/run_server.sh [mode] [port]

MODE=${1:-"federated"}
PORT=${2:-"8080"}

echo "Starting server in ${MODE} mode on port ${PORT}..."

if [ "$MODE" == "federated" ]; then
    python -m src.server.flower_server \
        --host "0.0.0.0" \
        --port "${PORT}" \
        --rounds 10 \
        --min-clients 2
else
    python -m src.server.main \
        --mode "${MODE}" \
        --host "0.0.0.0" \
        --port "${PORT}" \
        --config "configs/server_config.yaml"
fi
