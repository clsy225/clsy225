#!/usr/bin/env bash
set -euo pipefail

echo "=== host devices ==="
ls -l /dev/ttyUSB0 /dev/video0 /dev/video1 2>/dev/null || true

echo

echo "=== container devices ==="
docker exec ros2-pi bash -lc 'ls -l /dev/ttyUSB0 /dev/video0 /dev/video1'

echo

echo "=== ros env ==="
docker exec ros2-pi bash -lc 'source /opt/ros/humble/setup.bash && printenv | grep -E "ROS_|RMW_" | sort'

echo

echo "=== topics ==="
docker exec ros2-pi bash -lc 'source /opt/ros/humble/setup.bash && ros2 topic list || true'
