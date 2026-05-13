#!/usr/bin/env bash
set -euo pipefail

docker exec -it ros2-pi bash -lc '
source /opt/ros/humble/setup.bash
ros2 launch slam_toolbox online_async_launch.py base_frame:=base_footprint odom_frame:=base_footprint
'
