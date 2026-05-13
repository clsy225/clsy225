#!/usr/bin/env bash
set -euo pipefail

cd /home/pi/.openclaw/workspace/ros2_docker
./save_map_png.sh
./copy_map_png_to_host.sh
./save_camera_png.sh
./copy_camera_png_to_host.sh

echo
echo 'Exported:'
ls -lh /home/pi/.openclaw/workspace/map_snapshot.png /home/pi/.openclaw/workspace/camera_snapshot.png
