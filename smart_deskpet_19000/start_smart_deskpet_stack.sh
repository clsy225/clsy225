#!/usr/bin/env bash
set -euo pipefail

cd /userdata
mkdir -p /home/linaro/.openclaw/logs

stop_port() {
  local port="$1"
  local pids
  pids="$(ss -ltnp "sport = :${port}" 2>/dev/null | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | sort -u)"
  if [ -n "$pids" ]; then
    kill $pids 2>/dev/null || true
    sleep 1
    kill -9 $pids 2>/dev/null || true
  fi
}

pkill -f 'tts5_web.py' || true
pkill -f 'live2d_test_server.py' || true
pkill -f 'live2d_gesture_merged.py' || true
pkill -f 'smart_deskpet_dashboard_19000.py' || true
stop_port 5002
stop_port 8095
stop_port 19000

nohup python3 /userdata/openclaw_workspace/live2d_gesture_merged.py > /home/linaro/.openclaw/logs/live2d_gesture_merged.log 2>&1 &
echo $! > /home/linaro/.openclaw/logs/live2d_gesture_merged.pid

nohup python3 /userdata/tts5_web.py > /home/linaro/.openclaw/logs/tts5_web_userdata.log 2>&1 &
echo $! > /home/linaro/.openclaw/logs/tts5_web_userdata.pid

if ! ss -ltnp | grep -q ':9880\b'; then
  nohup bash -lc 'cd /userdata/rknn_voice_test && python api_rknn_keepalive.py -g "/home/linaro/GPT_weights_v2Pro/ldnn-e15.ckpt" -s "/home/linaro/SoVITS_weights_v2Pro/xxx_e8_s640.pth" -a 0.0.0.0 -p 9880 -dr "/home/linaro/witch.wav" -dt "魔女" -dl "ja"' > /home/linaro/.openclaw/logs/rknn_voice_9880.log 2>&1 &
  echo $! > /home/linaro/.openclaw/logs/rknn_voice_9880.pid
fi

nohup python3 /userdata/smart_deskpet_dashboard_19000.py > /home/linaro/.openclaw/logs/smart_deskpet_dashboard_19000.log 2>&1 &
echo $! > /home/linaro/.openclaw/logs/smart_deskpet_dashboard_19000.pid

sleep 3
curl -s -X POST http://127.0.0.1:19000/api/start >/dev/null 2>&1 || true
sleep 2

echo '=== ports ==='
ss -ltnp | egrep ':(5002|8095|9880|19000|8088)\b' || true

echo '=== pids ==='
echo 'live2d pid: '$(cat /home/linaro/.openclaw/logs/live2d_gesture_merged.pid)
echo 'tts5 pid: '$(cat /home/linaro/.openclaw/logs/tts5_web_userdata.pid)
echo 'dashboard pid: '$(cat /home/linaro/.openclaw/logs/smart_deskpet_dashboard_19000.pid)
[ -f /home/linaro/.openclaw/logs/rknn_voice_9880.pid ] && echo 'rknn_voice pid: '$(cat /home/linaro/.openclaw/logs/rknn_voice_9880.pid) || true

echo '=== running ==='
pgrep -af 'live2d_gesture_merged.py|tts5_web.py|smart_deskpet_dashboard_19000.py|api_rknn_keepalive.py|api.py.*9880|GPTSoVITS|zongtitrack4.py' || true
