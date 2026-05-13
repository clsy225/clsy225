# ROS2 Docker on Raspberry Pi 5 (Debian 13)

This directory sets up a practical ROS2 container for:
- YDLIDAR X3 Pro on `/dev/ttyUSB0`
- USB camera on `/dev/video0` (and `/dev/video1` exposed too)
- future `/scan`, `/image_raw`, and `slam_toolbox`

## What this gives you now

- A Dockerized ROS2 Humble shell on the Pi
- Host device passthrough for the LiDAR and USB camera
- `usb_cam`, `slam_toolbox`, `cv_bridge`, `image_transport` installed in the container
- A mounted app workspace at `./app`

## Important reality check

This uses `arm64v8/ros:humble-ros-base` as the base image.
That is the fastest path, but image/package availability can vary over time.
If a package/image pull fails, switch to a custom Ubuntu 22.04 base and install ROS2 manually inside the Dockerfile.

## Start

```bash
cd /home/pi/.openclaw/workspace/ros2_docker
chmod +x start_ros2.sh check_ros2.sh
./start_ros2.sh
```

## Verify devices/topics

```bash
cd /home/pi/.openclaw/workspace/ros2_docker
./check_ros2.sh
```

## Inside the container

### Source ROS2
```bash
source /opt/ros/humble/setup.bash
```

### Build mounted workspace
```bash
cd /root/ws
colcon build --symlink-install
source install/setup.bash
```

### Check camera
```bash
v4l2-ctl --list-devices
ros2 run usb_cam usb_cam_node_exe --ros-args -p video_device:=/dev/video0 -p image_width:=640 -p image_height:=480 -p pixel_format:=mjpeg2rgb
```

### Check LiDAR script access from container
```bash
python3 - <<'PY'
import serial
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
print('opened lidar', ser.port)
ser.close()
PY
```

## Bridging your current LiDAR parser into ROS2

You already have a host-side script:
- `/home/pi/.openclaw/workspace/ydlidar_ros2_scan.py`

Next step is to copy/adapt it into `app/` as a proper ROS2 Python package, then run it in the container so it publishes `/scan`.

## Suggested order

1. Start container successfully
2. Verify `/dev/ttyUSB0` and `/dev/video0` inside container
3. Start `usb_cam` and verify `/image_raw`
4. Turn the YDLIDAR ROS2 script into a package and publish `/scan`
5. Run `slam_toolbox`

## SLAM after topics work

Example direction:

```bash
source /opt/ros/humble/setup.bash
ros2 launch slam_toolbox online_async_launch.py
```

This assumes `/scan` is being published correctly.

## Files

- `docker-compose.yml` — container/runtime wiring
- `Dockerfile` — ROS2 image with camera/SLAM deps
- `start_ros2.sh` — build/start and shell into container
- `check_ros2.sh` — verify devices and ROS env
- `app/` — mounted source workspace for your ROS2 packages

## Notes

- Container uses `network_mode: host` for simpler ROS2 DDS discovery.
- `privileged: true` is used here for simplicity while bringing hardware up.
- Once stable, you can tighten device permissions and reduce privileges.
