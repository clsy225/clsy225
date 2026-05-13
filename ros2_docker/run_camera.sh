#!/usr/bin/env bash
set -euo pipefail

docker exec -it ros2-pi bash -lc '
source /opt/ros/humble/setup.bash
ros2 run usb_cam usb_cam_node_exe --ros-args \
  -p video_device:=/dev/video0 \
  -p image_width:=640 \
  -p image_height:=480 \
  -p pixel_format:=mjpeg2rgb
'
