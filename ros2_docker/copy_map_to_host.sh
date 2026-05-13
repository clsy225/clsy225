#!/usr/bin/env bash
set -euo pipefail

docker cp ros2-pi:/root/ws/mymap.pgm /home/pi/.openclaw/workspace/mymap.pgm
docker cp ros2-pi:/root/ws/mymap.yaml /home/pi/.openclaw/workspace/mymap.yaml
ls -lh /home/pi/.openclaw/workspace/mymap.*
