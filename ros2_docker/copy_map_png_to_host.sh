#!/usr/bin/env bash
set -euo pipefail

docker cp ros2-pi:/root/ws/map_snapshot.png /home/pi/.openclaw/workspace/map_snapshot.png
docker cp ros2-pi:/root/ws/map_snapshot.txt /home/pi/.openclaw/workspace/map_snapshot.txt
ls -lh /home/pi/.openclaw/workspace/map_snapshot.*
