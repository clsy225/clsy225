#!/usr/bin/env bash
set -euo pipefail
cd /home/linaro/.openclaw/workspace
mkdir -p /home/linaro/.openclaw/logs
pkill -f 'python3 /home/linaro/.openclaw/workspace/tts5_web.py' || true
nohup python3 /home/linaro/.openclaw/workspace/tts5_web.py > /home/linaro/.openclaw/logs/tts5_web.log 2>&1 &
echo $! > /home/linaro/.openclaw/logs/tts5_web.pid
sleep 2
ss -ltnp | grep 8095 || true
echo 'log: /home/linaro/.openclaw/logs/tts5_web.log'
echo 'pid: '$(cat /home/linaro/.openclaw/logs/tts5_web.pid)
