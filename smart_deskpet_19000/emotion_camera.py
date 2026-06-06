import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
import time
import threading
import queue
import os
import socket
import struct

# ===================== 配置 =====================
WEB_PORT = 8088
UPLOAD_INTERVAL = 1.0          # 每隔多少秒发一张到上位机
JPEG_QUALITY = 85
REMOTE_SERVER_IP = os.environ.get('EMOTION_REMOTE_IP', '192.168.137.1')
REMOTE_SERVER_PORT = int(os.environ.get('EMOTION_REMOTE_PORT', '9999'))
SOCKET_TIMEOUT = float(os.environ.get('EMOTION_SOCKET_TIMEOUT', '10'))


def camera_device_candidates(defaults=None):
    candidates = []

    def add(value):
        try:
            if isinstance(value, str) and value.startswith('/dev/video-camera0') and os.path.exists(value):
                value = os.path.realpath(value)
            if isinstance(value, str) and value.startswith('/dev/video'):
                value = value.rsplit('video', 1)[1]
            value = int(value)
            if value not in candidates and os.path.exists(f'/dev/video{value}'):
                candidates.append(value)
        except Exception:
            pass

    for env_name in ('SMART_CAMERA_DEVICE', 'CAMERA_DEVICE'):
        env_value = os.environ.get(env_name)
        if env_value:
            add(env_value)
    add('/dev/video-camera0')
    for value in (63, 64, 62, 0, 1, 53, 54):
        add(value)
    for name in os.listdir('/dev') if os.path.isdir('/dev') else []:
        if name.startswith('video') and name[5:].isdigit():
            add(name[5:])
    for value in defaults or []:
        add(value)
    return candidates or list(defaults or [0])


CAMERA_DEVICE_CANDIDATES = camera_device_candidates([62])

# ===================== 全局 =====================
app = Flask(__name__)
camera = None
current_emotion = '等待中'
current_confidence = 0.0
last_detect_time = 0.0
thread_lock = threading.Lock()
frame_queue = queue.Queue(maxsize=2)
latest_frame = None
stop_thread = False
camera_thread_started = False
last_remote_raw = ''
last_send_bytes = 0
last_error = ''

# ===================== 摄像头初始化 =====================
def init_camera_device():
    global camera
    print('正在打开摄像头...')
    for dev_id in CAMERA_DEVICE_CANDIDATES:
        try:
            camera = cv2.VideoCapture(dev_id, cv2.CAP_V4L2)
            if camera.isOpened():
                ret, _ = camera.read()
                if ret:
                    print(f'✅ 摄像头打开成功 /dev/video{dev_id}')
                    return True
                camera.release()
        except Exception:
            pass
    print('❌ 摄像头打开失败')
    return False

# ===================== 远程情绪TCP =====================
def send_frame_to_remote(frame):
    global current_emotion, current_confidence, last_remote_raw, last_send_bytes, last_error

    ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError('JPEG 编码失败')

    img_bytes = encoded.tobytes()
    last_send_bytes = len(img_bytes)

    with socket.create_connection((REMOTE_SERVER_IP, REMOTE_SERVER_PORT), timeout=SOCKET_TIMEOUT) as sock:
        sock.settimeout(SOCKET_TIMEOUT)
        sock.sendall(struct.pack('!I', len(img_bytes)))
        sock.sendall(img_bytes)

        chunks = []
        while True:
            try:
                chunk = sock.recv(4096)
            except socket.timeout:
                break
            if not chunk:
                break
            chunks.append(chunk)

    response = b''.join(chunks).decode('utf-8', errors='replace').strip()
    last_remote_raw = response
    last_error = ''
    print(f'🎭 上位机返回: {response}', flush=True)

    if response.startswith('ERROR:'):
        raise RuntimeError(response)

    # 协议预期: "情绪,0.9876"
    emotion = response
    confidence = 0.0
    if ',' in response:
        parts = response.split(',', 1)
        emotion = parts[0].strip()
        try:
            confidence = float(parts[1].strip())
        except Exception:
            confidence = 0.0

    with thread_lock:
        current_emotion = emotion
        current_confidence = confidence

    return {
        'emotion': emotion,
        'confidence': confidence,
        'raw': response,
        'bytes_sent': len(img_bytes),
    }

# ===================== 摄像头线程 =====================
def camera_reader():
    global stop_thread, latest_frame
    while not stop_thread:
        if camera and camera.isOpened():
            ret, frame = camera.read()
            if ret:
                latest_frame = frame.copy()
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                frame_queue.put(frame)
        time.sleep(0.005)


def ensure_camera_thread():
    global camera_thread_started
    if not camera_thread_started:
        threading.Thread(target=camera_reader, daemon=True).start()
        camera_thread_started = True

# ===================== 视频流 =====================
def generate_video_stream():
    global last_detect_time, last_error, latest_frame
    ensure_camera_thread()

    while True:
        frame = latest_frame.copy() if latest_frame is not None else None
        if frame is None:
            time.sleep(0.05)
            continue

        now = time.time()
        if now - last_detect_time >= UPLOAD_INTERVAL:
            try:
                send_frame_to_remote(frame)
            except Exception as e:
                last_error = str(e)
                print(f'❌ 远程情绪识别失败: {e}', flush=True)
                with thread_lock:
                    current_emotion = 'Error'
                    current_confidence = 0.0
            last_detect_time = now

        display = frame.copy()
        with thread_lock:
            emotion = current_emotion
            conf = current_confidence
        text = f'Emotion: {emotion} ({conf:.2f})'
        cv2.putText(display, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(display, f'Remote: {REMOTE_SERVER_IP}:{REMOTE_SERVER_PORT}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
        if last_error:
            cv2.putText(display, f'ERR: {last_error[:60]}', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        ok, jpeg = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            time.sleep(0.05)
            continue
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'
        time.sleep(0.03)

# ===================== 页面 =====================
@app.route('/')
def index_page():
    return f'''
<html>
<head>
    <meta charset="UTF-8">
    <title>远程表情识别系统</title>
    <style>
        body{{background:#1a1a2e;text-align:center;color:white;font-family:Arial;padding:20px;}}
        img{{border:5px solid #00ff88;border-radius:10px;max-width:90vw;}}
        .card{{display:inline-block;background:#111827;border-radius:14px;padding:16px 20px;margin:12px;min-width:320px;}}
        .muted{{color:#aeb7c4;font-size:14px;}}
        .btn{{background:#00ff88;color:#000;padding:12px 28px;font-size:18px;border:none;border-radius:10px;cursor:pointer;margin:10px;}}
    </style>
</head>
<body>
    <h1>😊 持续发送图片到上位机做情绪识别</h1>
    <div class="card">
      <div>上位机：<b>{REMOTE_SERVER_IP}:{REMOTE_SERVER_PORT}</b></div>
      <div class="muted">每 {UPLOAD_INTERVAL:.1f} 秒发送一张 JPEG 图片</div>
      <div class="muted">当前页面使用快照刷新模式，兼容性比 MJPEG 视频流更好</div>
      <button class="btn" onclick="manualOnce()">立即发送一张</button>
      <pre id="result">等待结果...</pre>
    </div>
    <br>
    <img id="cam" src="/snapshot.jpg" width="960">

    <script>
      const cam = document.getElementById('cam');

      async function refreshState() {{
        const r = await fetch('/api/state');
        const j = await r.json();
        document.getElementById('result').textContent = JSON.stringify(j, null, 2);
      }}

      async function manualOnce() {{
        const r = await fetch('/api/send_once', {{ method: 'POST' }});
        const j = await r.json();
        document.getElementById('result').textContent = JSON.stringify(j, null, 2);
      }}

      function refreshSnapshot() {{
        cam.src = '/snapshot.jpg?t=' + Date.now();
      }}

      setInterval(refreshState, 1000);
      setInterval(refreshSnapshot, 200);
      refreshState();
      refreshSnapshot();
    </script>
</body>
</html>
'''

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/snapshot.jpg')
def snapshot_jpg():
    ensure_camera_thread()
    frame = latest_frame.copy() if latest_frame is not None else None
    if frame is None:
        return jsonify({'ok': False, 'error': '暂无摄像头帧'}), 503
    display = frame.copy()
    with thread_lock:
        emotion = current_emotion
        conf = current_confidence
    cv2.putText(display, f'Emotion: {emotion} ({conf:.2f})', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    ok, jpeg = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        return jsonify({'ok': False, 'error': 'JPEG 编码失败'}), 500
    return Response(jpeg.tobytes(), mimetype='image/jpeg')

@app.route('/api/state')
def api_state():
    with thread_lock:
        emotion = current_emotion
        conf = current_confidence
    return jsonify({
        'ok': True,
        'emotion': emotion,
        'confidence': conf,
        'remote_ip': REMOTE_SERVER_IP,
        'remote_port': REMOTE_SERVER_PORT,
        'upload_interval': UPLOAD_INTERVAL,
        'last_remote_raw': last_remote_raw,
        'last_send_bytes': last_send_bytes,
        'last_error': last_error,
    })

@app.route('/api/send_once', methods=['POST'])
def api_send_once():
    ensure_camera_thread()
    try:
        frame = frame_queue.get(timeout=1.0)
    except queue.Empty:
        return jsonify({'ok': False, 'error': '暂无摄像头帧'}), 500
    try:
        result = send_frame_to_remote(frame)
        return jsonify({'ok': True, **result})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e), 'last_remote_raw': last_remote_raw}), 500

# ===================== 主程序 =====================
if __name__ == '__main__':
    print('=' * 60)
    print('  远程表情识别板端启动中')
    print('=' * 60)
    if not init_camera_device():
        raise SystemExit('摄像头初始化失败')
    ensure_camera_thread()
    print(f'🌐 页面地址：http://0.0.0.0:{WEB_PORT}')
    print(f'🎯 上位机地址：{REMOTE_SERVER_IP}:{REMOTE_SERVER_PORT}')
    print(f'🖼️ 持续发送间隔：{UPLOAD_INTERVAL:.1f} 秒/张')
    print('=' * 60)
    try:
        app.run(host='0.0.0.0', port=WEB_PORT, debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        stop_thread = True
    finally:
        if camera:
            camera.release()
        print('✅ 程序已退出')
