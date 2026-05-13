#!/usr/bin/env bash
set -euo pipefail

# Restart container-side runtime components in a clean order.
# Host-side wrapper intentionally avoids TTY allocation so it can run under nohup.

docker exec ros2-pi bash -lc 'pkill -f ydlidar_ros2_scan.py || true; pkill -f static_transform_publisher || true; pkill -f slam_toolbox || true; pkill -f usb_cam_node_exe || true'
sleep 2

nohup /home/pi/.openclaw/workspace/ros2_docker/run_lidar_scan.sh >/home/pi/.openclaw/workspace/lidar.log 2>&1 &
nohup /home/pi/.openclaw/workspace/ros2_docker/run_static_tf.sh >/home/pi/.openclaw/workspace/static_tf.log 2>&1 &
nohup /home/pi/.openclaw/workspace/ros2_docker/run_slam_direct.sh >/home/pi/.openclaw/workspace/slam.log 2>&1 &
nohup /home/pi/.openclaw/workspace/ros2_docker/run_camera.sh >/home/pi/.openclaw/workspace/camera.log 2>&1 &

sleep 8

echo '=== topics ==='
docker exec ros2-pi bash -lc "source /opt/ros/humble/setup.bash && ros2 topic list | grep -E '/map|/scan|/image_raw|/camera_info' || true"

echo '=== slam params ==='
docker exec ros2-pi bash -lc "source /opt/ros/humble/setup.bash && ros2 param get /slam_toolbox base_frame && ros2 param get /slam_toolbox odom_frame || true"

echo '=== recent logs ==='
for f in /home/pi/.openclaw/workspace/lidar.log /home/pi/.openclaw/workspace/static_tf.log /home/pi/.openclaw/workspace/slam.log /home/pi/.openclaw/workspace/camera.log; do
  echo "--- $f ---"
  tail -n 40 "$f" || true
  echo
 done
