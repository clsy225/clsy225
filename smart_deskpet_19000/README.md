# Smart Deskpet Dashboard 19000

This folder contains the RK3588 `/userdata` dashboard code used by the smart deskpet control panel.

## Main Entry

- `smart_deskpet_dashboard_19000.py`: Flask dashboard on port `19000`
- `start_smart_deskpet_stack.sh`: helper script for starting dashboard-related services
- `openclaw_workspace/ai_engine.py`: local/cloud AI engine wrapper
- `openclaw_workspace/live2d_gesture_merged.py`: Live2D + gesture camera service
- `rknn_voice_test/api_rknn_keepalive.py`: RKNN GPT-SoVITS keepalive API for port `9880`
- `tts4.py`: legacy voice command assistant, sanitized to read `DEEPSEEK_API_KEY` from env
- `tts5_web.py`: voiceprint web service
- `emotion_camera.py`: emotion camera service
- `userdata_services/`: visual services launched by the 19000 dashboard, including gesture, total tracking, face/head detection, mouse control, and legacy voice assistants
- `openclaw_workspace_services/`: direct 8080/8083 launcher dependencies, including RKNN camera and Ningning integrated browser helpers

## Deploy

Copy files back to the board under `/userdata` and restart:

```bash
cd /userdata
bash /userdata/start_smart_deskpet_stack.sh
```

Open:

```text
http://192.168.137.101:19000
```

## RKNN TTS

The optimized local TTS service is expected at `127.0.0.1:9880`:

```bash
cd /userdata/rknn_voice_test
python api_rknn_keepalive.py \
  -g /home/linaro/GPT_weights_v2Pro/ldnn-e15.ckpt \
  -s /home/linaro/SoVITS_weights_v2Pro/xxx_e8_s640.pth \
  -a 0.0.0.0 \
  -p 9880 \
  -dr /home/linaro/witch.wav \
  -dt "魔女" \
  -dl ja
```

Model weights, logs, cache files, and generated audio are not included.
