#!/bin/bash
# Run edge node
# Usage: ./scripts/run_edge.sh [node_id] [server_ip] [strategy]

NODE_ID=${1:-"A"}
SERVER_IP=${2:-"192.168.1.100"}
STRATEGY=${3:-"federated"}

echo "Starting edge node ${NODE_ID}..."
echo "Server: ${SERVER_IP}"
echo "Strategy: ${STRATEGY}"

python -m src.edge.main \
    --node-id "${NODE_ID}" \
    --server-ip "${SERVER_IP}" \
    --strategy "${STRATEGY}" \
    --config "configs/edge_config.yaml"
