#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export DISPLAY=${DISPLAY:-:0}

echo "[1/3] Building/starting ROS2 container..."
docker-compose up -d --build

echo "[2/3] Container status:"
docker ps --filter name=ros2-pi

echo "[3/3] Opening shell in container..."
docker exec -it ros2-pi bash
