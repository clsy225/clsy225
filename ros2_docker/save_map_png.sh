#!/usr/bin/env bash
set -euo pipefail

docker exec -it ros2-pi bash -lc '
source /opt/ros/humble/setup.bash
python3 /root/ws/src/app/map_to_png.py /root/ws/map_snapshot.png 15
'
