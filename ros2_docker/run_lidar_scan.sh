#!/usr/bin/env bash
set -euo pipefail

docker exec -it ros2-pi bash -lc '
source /opt/ros/humble/setup.bash
python3 /root/ws/src/app/ydlidar_ros2_scan.py /dev/ttyUSB0 115200 laser_frame
'
