#!/usr/bin/env bash
set -euo pipefail

docker exec ros2-pi bash -lc '
source /opt/ros/humble/setup.bash
ros2 run slam_toolbox async_slam_toolbox_node \
  --ros-args \
  -p base_frame:=base_footprint \
  -p odom_frame:=base_footprint \
  -p map_frame:=map \
  -p scan_topic:=/scan \
  -p use_scan_matching:=true
'
