#!/usr/bin/env bash
set -euo pipefail

docker exec ros2-pi bash -lc '
source /opt/ros/humble/setup.bash
python3 /root/ws/src/app/camera_to_png.py /root/ws/camera_snapshot.png 15 /image_raw
'
