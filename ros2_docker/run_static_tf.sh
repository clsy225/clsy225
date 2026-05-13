#!/usr/bin/env bash
set -euo pipefail

docker exec ros2-pi bash -lc '
source /opt/ros/humble/setup.bash
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 base_footprint laser_frame
'
