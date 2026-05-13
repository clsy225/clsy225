#!/usr/bin/env bash
set -euo pipefail

docker cp ros2-pi:/root/ws/camera_snapshot.png /home/pi/.openclaw/workspace/camera_snapshot.png
ls -lh /home/pi/.openclaw/workspace/camera_snapshot.png
