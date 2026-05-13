#!/usr/bin/env bash
set -euo pipefail

docker exec -it ros2-pi bash -lc '
source /opt/ros/humble/setup.bash
ros2 run nav2_map_server map_saver_cli -f /root/ws/mymap
'
