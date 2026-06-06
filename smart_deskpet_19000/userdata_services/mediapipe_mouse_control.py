import math
import os
import queue
import subprocess
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, request, jsonify
from PIL import Image, ImageDraw, ImageFont

# ===================== 配置 =====================
WEB_PORT = 5001
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

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

# 鼠标调节参数（可网页实时修改）
SMOOTHING = 0.35
CLICK_DISTANCE = 35
DRAG_HOLD_SECONDS = 0.5
SENSITIVITY_X = 1.8
SENSITIVITY_Y = 1.8
DEADZONE = 0.015

DISPLAY_ENV = os.environ.get("DISPLAY", ":0")
XAUTHORITY_CANDIDATES = [
    "/home/linaro/.Xauthority",
    "/var/run/lightdm/root/:0",
]

# ===================== 全局 =====================
app = Flask(__name__)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = None
camera = None
frame_queue = queue.Queue(maxsize=2)
stop_thread = False
thread_lock = threading.Lock()

current_status = "等待手势"
mouse_backend = "none"
screen_w = 1920
screen_h = 1080
smoothed_x = None
smoothed_y = None
last_camera_x = None
last_camera_y = None
pinch_start_time = None
last_click_time = 0
is_dragging = False


# ===================== 中文字体 =====================
def cv2_put_text_cn(img, text, pos, font_size=30, color=(0, 255, 0)):
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", font_size)
        draw.text(pos, text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return img


# ===================== 工具函数 =====================
def cal(p1, p2, w, h):
    x1, y1 = int(p1.x * w), int(p1.y * h)
    x2, y2 = int(p2.x * w), int(p2.y * h)
    return math.hypot(x1 - x2, y1 - y2)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def smooth_value(old, new, alpha):
    if old is None:
        return new
    return old * (1 - alpha) + new * alpha


def run_cmd(cmd):
    env = os.environ.copy()
    env["DISPLAY"] = DISPLAY_ENV
    for cand in XAUTHORITY_CANDIDATES:
        if os.path.exists(cand):
            env["XAUTHORITY"] = cand
            break
    try:
        subprocess.run(cmd, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def detect_mouse_backend():
    global mouse_backend, screen_w, screen_h
    if subprocess.run(["which", "xdotool"], capture_output=True).returncode == 0:
        mouse_backend = "xdotool"
        try:
            env = os.environ.copy()
            env["DISPLAY"] = DISPLAY_ENV
            for cand in XAUTHORITY_CANDIDATES:
                if os.path.exists(cand):
                    env["XAUTHORITY"] = cand
                    break
            out = subprocess.check_output(["xdotool", "getdisplaygeometry"], env=env, text=True).strip()
            sw, sh = out.split()
            screen_w, screen_h = int(sw), int(sh)
        except Exception:
            screen_w, screen_h = 1920, 1080
        return
    mouse_backend = "none"
    screen_w, screen_h = 1920, 1080


def move_mouse(x, y):
    if mouse_backend == "xdotool":
        return run_cmd(["xdotool", "mousemove", str(int(x)), str(int(y))])
    return False


def mouse_down():
    if mouse_backend == "xdotool":
        return run_cmd(["xdotool", "mousedown", "1"])
    return False


def mouse_up():
    if mouse_backend == "xdotool":
        return run_cmd(["xdotool", "mouseup", "1"])
    return False


def mouse_click():
    if mouse_backend == "xdotool":
        return run_cmd(["xdotool", "click", "1"])
    return False


# ===================== 初始化 =====================
def init_mediapipe():
    global hands
    hands = mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=1,
    )
    print("✅ MediaPipe 初始化完成")


def init_camera_device():
    global camera
    for device in CAMERA_DEVICE_CANDIDATES:
        try:
            camera = cv2.VideoCapture(device, cv2.CAP_V4L2)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            if camera.isOpened():
                ret, _ = camera.read()
                if ret:
                    print(f"✅ 摄像头打开成功: /dev/video{device}")
                    return True
                camera.release()
        except Exception:
            pass
    print(f"❌ 摄像头打开失败，已尝试: {CAMERA_DEVICE_CANDIDATES}")
    return False


# ===================== 摄像头线程 =====================
def camera_reader():
    global stop_thread
    while not stop_thread:
        if camera and camera.isOpened():
            ret, frame = camera.read()
            if ret:
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except Exception:
                        pass
                frame_queue.put(frame)
        time.sleep(0.005)


# ===================== 手势转鼠标 =====================
def process_hand_mouse(frame):
    global current_status, smoothed_x, smoothed_y, last_camera_x, last_camera_y
    global pinch_start_time, last_click_time, is_dragging
    global SENSITIVITY_X, SENSITIVITY_Y, SMOOTHING, DEADZONE, CLICK_DISTANCE, DRAG_HOLD_SECONDS

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    status = f"后端: {mouse_backend}"

    if res.multi_hand_landmarks:
        hand = res.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        lm = hand.landmark

        index_tip = lm[8]
        thumb_tip = lm[4]
        middle_tip = lm[12]

        cam_x = 1.0 - index_tip.x
        cam_y = index_tip.y

        if last_camera_x is None:
            last_camera_x = cam_x
            last_camera_y = cam_y

        dx = cam_x - last_camera_x
        dy = cam_y - last_camera_y

        if abs(dx) < DEADZONE:
            dx = 0.0
        if abs(dy) < DEADZONE:
            dy = 0.0

        base_x = smoothed_x if smoothed_x is not None else cam_x * screen_w
        base_y = smoothed_y if smoothed_y is not None else cam_y * screen_h

        target_x = base_x + dx * screen_w * SENSITIVITY_X
        target_y = base_y + dy * screen_h * SENSITIVITY_Y

        target_x = clamp(target_x, 0, screen_w - 1)
        target_y = clamp(target_y, 0, screen_h - 1)

        smoothed_x = smooth_value(smoothed_x, target_x, SMOOTHING)
        smoothed_y = smooth_value(smoothed_y, target_y, SMOOTHING)
        last_camera_x = cam_x
        last_camera_y = cam_y
        move_mouse(smoothed_x, smoothed_y)

        pinch_dist = cal(index_tip, thumb_tip, w, h)
        two_finger_dist = cal(index_tip, middle_tip, w, h)

        if pinch_dist < CLICK_DISTANCE:
            if pinch_start_time is None:
                pinch_start_time = time.time()
            held = time.time() - pinch_start_time

            if held >= DRAG_HOLD_SECONDS and not is_dragging:
                mouse_down()
                is_dragging = True
                status = f"拖拽中 ({int(smoothed_x)}, {int(smoothed_y)})"
            elif not is_dragging:
                status = f"准备点击 ({int(smoothed_x)}, {int(smoothed_y)})"
        else:
            if pinch_start_time is not None:
                held = time.time() - pinch_start_time
                if is_dragging:
                    mouse_up()
                    is_dragging = False
                    status = "拖拽结束"
                elif held < DRAG_HOLD_SECONDS and time.time() - last_click_time > 0.4:
                    mouse_click()
                    last_click_time = time.time()
                    status = "单击"
            pinch_start_time = None

        if two_finger_dist > 80 and pinch_dist >= CLICK_DISTANCE and not is_dragging:
            status = f"移动光标 ({int(smoothed_x)}, {int(smoothed_y)})"

        ix, iy = int(index_tip.x * w), int(index_tip.y * h)
        tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
        cv2.circle(frame, (ix, iy), 10, (0, 255, 255), -1)
        cv2.circle(frame, (tx, ty), 10, (255, 0, 255), -1)
        cv2.line(frame, (ix, iy), (tx, ty), (255, 255, 0), 2)
    else:
        if is_dragging:
            mouse_up()
            is_dragging = False
        pinch_start_time = None
        last_camera_x = None
        last_camera_y = None
        status = f"未检测到手 | 后端: {mouse_backend}"

    current_status = status
    frame = cv2_put_text_cn(frame, current_status, (20, 40), 32, (0, 255, 0))
    frame = cv2_put_text_cn(frame, f"屏幕: {screen_w}x{screen_h}", (20, 80), 24, (255, 255, 0))
    frame = cv2_put_text_cn(frame, f"灵敏度X/Y: {SENSITIVITY_X:.2f}/{SENSITIVITY_Y:.2f} 平滑:{SMOOTHING:.2f} 死区:{DEADZONE:.3f}", (20, 115), 22, (255, 200, 0))
    frame = cv2_put_text_cn(frame, f"点击阈值:{CLICK_DISTANCE} 拖拽延时:{DRAG_HOLD_SECONDS:.2f}s", (20, 145), 22, (255, 200, 0))
    return frame


# ===================== 视频流 =====================
def generate_video_stream():
    threading.Thread(target=camera_reader, daemon=True).start()
    while True:
        try:
            frame = frame_queue.get(timeout=0.5)
        except Exception:
            continue
        with thread_lock:
            frame = process_hand_mouse(frame)
        _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n'


# ===================== 网页 =====================
HTML_TEMPLATE = '''
<html>
<head>
<meta charset="UTF-8">
<title>手势鼠标控制</title>
<style>
body{{background:#101820;color:white;text-align:center;font-family:sans-serif}}
img{{border:4px solid #00ff88;border-radius:10px}}
.card{{margin:20px auto;padding:12px;background:#1d2630;width:760px;border-radius:12px}}
label{{display:block;margin:8px 0}}
input{{width:320px}}
button{{padding:8px 16px;border-radius:8px;border:none;background:#00cc88;color:white;cursor:pointer}}
.small{{color:#9ad}}
</style>
</head>
<body>
<h1>🖐️ MediaPipe 手势鼠标控制</h1>
<div class="card">
<p>后端：{mouse_backend} | DISPLAY：{display_env}</p>
<p>食指移动光标，拇指+食指捏合单击，长按进入拖拽</p>
<img src="/video_feed" width="640">
<div style="margin-top:16px">
<label>灵敏度 X：<input id="sx" type="range" min="0.2" max="4.0" step="0.05" value="{sx}"> <span id="sxv">{sx}</span></label>
<label>灵敏度 Y：<input id="sy" type="range" min="0.2" max="4.0" step="0.05" value="{sy}"> <span id="syv">{sy}</span></label>
<label>平滑度：<input id="sm" type="range" min="0.05" max="0.95" step="0.01" value="{sm}"> <span id="smv">{sm}</span></label>
<label>死区：<input id="dz" type="range" min="0.000" max="0.100" step="0.001" value="{dz}"> <span id="dzv">{dz}</span></label>
<label>点击阈值：<input id="cd" type="range" min="10" max="80" step="1" value="{cd}"> <span id="cdv">{cd}</span></label>
<label>拖拽延时：<input id="dh" type="range" min="0.10" max="1.50" step="0.05" value="{dh}"> <span id="dhv">{dh}</span></label>
<button onclick="saveCfg()">保存参数</button>
<p class="small">灵敏度越高越快，平滑越高越稳，死区越大越防抖。</p>
</div>
</div>
<script>
function bind(id){{
  const el = document.getElementById(id);
  const out = document.getElementById(id + 'v');
  el.addEventListener('input', () => {{ out.textContent = el.value; }});
}}
['sx','sy','sm','dz','cd','dh'].forEach(bind);
async function saveCfg(){{
  const body = {{
    sensitivity_x: parseFloat(document.getElementById('sx').value),
    sensitivity_y: parseFloat(document.getElementById('sy').value),
    smoothing: parseFloat(document.getElementById('sm').value),
    deadzone: parseFloat(document.getElementById('dz').value),
    click_distance: parseInt(document.getElementById('cd').value),
    drag_hold_seconds: parseFloat(document.getElementById('dh').value)
  }};
  const r = await fetch('/config', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify(body)
  }});
  const j = await r.json();
  alert('已更新: ' + JSON.stringify(j));
}}
</script>
</body>
</html>
'''


@app.route('/')
def index():
    return HTML_TEMPLATE.format(
        mouse_backend=mouse_backend,
        display_env=DISPLAY_ENV,
        sx=SENSITIVITY_X,
        sy=SENSITIVITY_Y,
        sm=SMOOTHING,
        dz=DEADZONE,
        cd=CLICK_DISTANCE,
        dh=DRAG_HOLD_SECONDS,
    )


@app.route('/config', methods=['GET', 'POST'])
def config_api():
    global SENSITIVITY_X, SENSITIVITY_Y, SMOOTHING, DEADZONE, CLICK_DISTANCE, DRAG_HOLD_SECONDS
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        SENSITIVITY_X = float(data.get('sensitivity_x', SENSITIVITY_X))
        SENSITIVITY_Y = float(data.get('sensitivity_y', SENSITIVITY_Y))
        SMOOTHING = float(data.get('smoothing', SMOOTHING))
        DEADZONE = float(data.get('deadzone', DEADZONE))
        CLICK_DISTANCE = int(data.get('click_distance', CLICK_DISTANCE))
        DRAG_HOLD_SECONDS = float(data.get('drag_hold_seconds', DRAG_HOLD_SECONDS))
    return jsonify({
        'sensitivity_x': SENSITIVITY_X,
        'sensitivity_y': SENSITIVITY_Y,
        'smoothing': SMOOTHING,
        'deadzone': DEADZONE,
        'click_distance': CLICK_DISTANCE,
        'drag_hold_seconds': DRAG_HOLD_SECONDS,
        'mouse_backend': mouse_backend,
        'screen': [screen_w, screen_h],
    })


@app.route('/video_feed')
def feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ===================== 主程序 =====================
if __name__ == '__main__':
    print('=' * 60)
    print(' MediaPipe 手势鼠标控制系统 ')
    print('=' * 60)
    detect_mouse_backend()
    print(f'鼠标后端: {mouse_backend}')
    print(f'DISPLAY: {DISPLAY_ENV}')
    init_mediapipe()
    init_camera_device()
    print(f'🌐 访问：http://192.168.137.101:{WEB_PORT}')
    try:
        app.run(host='0.0.0.0', port=WEB_PORT, debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        stop_thread = True
    finally:
        if camera:
            camera.release()
        if hands:
            hands.close()
        print('✅ 已退出')
