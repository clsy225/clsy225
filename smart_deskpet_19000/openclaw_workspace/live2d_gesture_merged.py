from flask import Flask, send_from_directory, Response, jsonify, request
import json
import socket
import subprocess
import os
import cv2
import numpy as np
import mediapipe as mp
import math
import time
import threading
import queue
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)
PAGE_TELEMETRY = {"events": []}

# ===================== Live2D 配置 =====================
MODEL_DIR = "/userdata/丛雨live2d第二版（新增表情）"
RUNTIME_DIR = "/userdata/live2d_runtime"
MODEL_FILE = "Murasame.model3.json"
WEB_PORT = 5002

# ===================== 手势识别配置 =====================
DETECTION_INTERVAL = 0.05
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
GESTURE_COOLDOWN = 30.0
CAMERA_JPEG_QUALITY = 85

# ===================== 手势识别全局 =====================
hands = None
mp_drawing = None
mp_hands = None
camera = None
current_gesture = "普通手势"
last_detect_time = 0.0
thread_lock = threading.Lock()
frame_queue = queue.Queue(maxsize=2)
stop_thread = False
camera_thread_started = False
last_gesture_change_at = 0.0

GESTURE_MAP = {
    "普通手势": {"param": "ParamCheek42", "expression": None, "label": "正常"},
    "OK手势 ✔️": {"motion_group": "Tapxiongbu", "motion_index": 0, "label": "Tapxiongbu[0]"},
    "挥拳动作 ✊": {"motion_group": "Tapxiongbu", "motion_index": 1, "label": "Tapxiongbu[1] / motion10"},
    "握拳动作 👊": {"param": "ParamCheek13", "expression": 3, "label": "怒嘴+蓄力"},
    "大拇指动作 👍": {"motion_group": "Tapleg", "motion_index": 0, "label": "Tapleg[0] / motion09"},
    "yes手势 👌": {"motion_group": "Taphair", "motion_index": 0, "label": "Taphair[0]"},
    "挥手动作 👋": {"motion_group": "Tapface", "motion_index": 0, "label": "Tapface[0]"},
    "比心 ❤️": {"motion_group": "Tapleg", "motion_index": 1, "label": "Tapleg[1] / motion11"},
    "双手检测中...": {"param": "ParamCheek40", "expression": 5, "label": "双手互动"},
}


def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


HOST_IP = get_host_ip()

with open(f"{MODEL_DIR}/{MODEL_FILE}", "r", encoding="utf-8") as f:
    MODEL_JSON = json.load(f)

FILE_REFS = MODEL_JSON.get("FileReferences", {})
EXPRESSIONS = FILE_REFS.get("Expressions", [])
MOTIONS = FILE_REFS.get("Motions", {})
MOTION_GROUPS = []
for group_name, items in MOTIONS.items():
    safe_items = []
    for idx, item in enumerate(items):
        safe_items.append({
            "index": idx,
            "file": item.get("File", ""),
            "sound": item.get("Sound", ""),
            "text": item.get("Text", ""),
            "fade_in": item.get("FadeInTime"),
            "fade_out": item.get("FadeOutTime"),
            "interruptable": item.get("Interruptable"),
        })
    MOTION_GROUPS.append({"group": group_name, "items": safe_items})

# ===================== 中文字体 =====================
def cv2_put_text_cn(img, text, pos, font_size=40, color=(0, 255, 0)):
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", font_size)
        draw.text(pos, text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception:
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        return img


# ===================== 手势识别函数 =====================
def cal(p1, p2, w, h):
    x1 = int(p1.x * w)
    y1 = int(p1.y * h)
    x2 = int(p2.x * w)
    y2 = int(p2.y * h)
    return math.hypot(x1 - x2, y1 - y2)


def is_palm_facing(landmark):
    return abs(landmark[0].x - landmark[9].x) > 0.05


def OKfigure(landmark, wid, hig):
    tip1 = landmark[4]
    tip2 = landmark[8]
    tip3 = landmark[12]
    tip4 = landmark[16]
    tip5 = landmark[20]
    distance = cal(tip1, tip2, wid, hig)
    dis_true = distance < 50 and distance > 0
    if tip3.y < landmark[9].y and tip4.y < landmark[13].y and tip5.y < landmark[17].y:
        return dis_true
    return False


wave_history = []
WAVE_BUFFER_SIZE = 10


def wave_figure(landmark, wid, hig):
    global wave_history
    fingers_straight = (
        landmark[8].y < landmark[6].y and landmark[12].y < landmark[10].y
        and landmark[16].y < landmark[14].y and landmark[20].y < landmark[18].y
    )
    if not fingers_straight:
        wave_history.clear()
        return False
    x = int(landmark[0].x * wid)
    wave_history.append(x)
    if len(wave_history) > WAVE_BUFFER_SIZE:
        wave_history.pop(0)
    if len(wave_history) == 10:
        return max(wave_history) - min(wave_history) > 30
    return False


def is_fist(landmark):
    index_fist = landmark[8].y > landmark[6].y
    middle_fist = landmark[12].y > landmark[10].y
    ring_fist = landmark[16].y > landmark[14].y
    pinky_fist = landmark[20].y > landmark[18].y
    four_fist = index_fist and middle_fist and ring_fist and pinky_fist
    if not four_fist:
        return False
    thumb_bent = landmark[4].y > landmark[3].y
    thumb_side = abs(landmark[4].x - landmark[6].x) > 0.08 and landmark[4].y > landmark[8].y - 0.05
    return thumb_bent or thumb_side


def yes_figure(landmark, wid, hig):
    tip1, tip2, tip3, tip4, tip5 = landmark[4], landmark[8], landmark[12], landmark[16], landmark[20]
    tip6, tip7 = landmark[7], landmark[13]
    x1, x2, x3 = int(tip1.x * wid), int(tip4.x * wid), int(tip5.x * wid)
    y1, y2, y3 = int(tip1.y * hig), int(tip4.y * hig), int(tip5.y * hig)
    ok = abs(x1 - x2) < 40 and abs(x1 - x3) < 40 and abs(x2 - x3) < 40 and abs(y1 - y2) < 40
    return ok and tip2.y < tip6.y and tip3.y < tip7.y


fight_history = []


def fight(landmark, wid, hig):
    global fight_history
    if not is_fist(landmark):
        return False
    x = int(landmark[0].x * wid)
    fight_history.append(x)
    if len(fight_history) > 8:
        fight_history.pop(0)
    return len(fight_history) == 8 and (max(fight_history) - min(fight_history)) > 50


def zan_figure(landmark, wid, hig):
    four = (
        landmark[8].y > landmark[6].y and landmark[12].y > landmark[10].y
        and landmark[16].y > landmark[14].y and landmark[20].y > landmark[18].y
    )
    thumb_up = landmark[4].y < landmark[3].y and landmark[3].y < landmark[2].y
    highest = landmark[4].y < landmark[12].y - 0.1
    return four and thumb_up and highest


def love_figure(lm1, lm2, w, h):
    d1 = cal(lm1[4], lm2[4], w, h)
    d2 = cal(lm1[8], lm2[8], w, h)
    d3 = cal(lm1[4], lm2[8], w, h)
    d4 = cal(lm1[8], lm2[4], w, h)
    return (d1 < 120 and d2 < 120) or (d3 < 120 and d4 < 120)


# ===================== 初始化 =====================
def init_mediapipe():
    global hands, mp_drawing, mp_hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=2,
    )
    mp_drawing = mp.solutions.drawing_utils
    print("✅ MediaPipe 初始化完成")


def init_camera_device():
    global camera
    for device in CAMERA_DEVICE_CANDIDATES:
        try:
            camera = cv2.VideoCapture(device, cv2.CAP_V4L2)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if camera.isOpened():
                ret, _ = camera.read()
                if ret:
                    print(f"✅ 摄像头打开成功 /dev/video{device}")
                    return True
                camera.release()
        except Exception:
            pass
    print(f"❌ 摄像头打开失败，已尝试: {CAMERA_DEVICE_CANDIDATES}")
    return False


def camera_reader():
    global stop_thread
    while not stop_thread:
        if camera and camera.isOpened():
            ret, f = camera.read()
            if ret:
                if frame_queue.full():
                    frame_queue.get_nowait()
                frame_queue.put(f)
        time.sleep(0.005)


def ensure_camera_thread():
    global camera_thread_started
    if not camera_thread_started:
        threading.Thread(target=camera_reader, daemon=True).start()
        camera_thread_started = True


def update_current_gesture(new_gesture):
    global current_gesture, last_gesture_change_at
    if new_gesture != current_gesture:
        current_gesture = new_gesture
        last_gesture_change_at = time.time()


def run_gesture(frame):
    h, w, _ = frame.shape
    txt = "普通手势"
    res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if res.multi_hand_landmarks:
        n = len(res.multi_hand_landmarks)
        for hl in res.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
        if n == 1:
            lm = res.multi_hand_landmarks[0].landmark
            if OKfigure(lm, w, h):
                txt = "OK手势 ✔️"
            elif fight(lm, w, h):
                txt = "挥拳动作 ✊"
            elif is_fist(lm):
                txt = "握拳动作 👊"
            elif zan_figure(lm, w, h):
                txt = "大拇指动作 👍"
            elif yes_figure(lm, w, h):
                txt = "yes手势 👌"
            elif wave_figure(lm, w, h):
                txt = "挥手动作 👋"
        if n == 2:
            if love_figure(res.multi_hand_landmarks[0].landmark, res.multi_hand_landmarks[1].landmark, w, h):
                txt = "比心 ❤️"
            else:
                txt = "双手检测中..."
    update_current_gesture(txt)
    return frame, txt


def generate_video_stream():
    global last_detect_time
    ensure_camera_thread()
    while True:
        try:
            frame = frame_queue.get(timeout=0.5)
        except Exception:
            continue
        t = time.time()
        if t - last_detect_time > DETECTION_INTERVAL:
            with thread_lock:
                frame, _ = run_gesture(frame)
            last_detect_time = t
        frame = cv2_put_text_cn(frame, current_gesture, (20, 50), 40, (0, 255, 0))
        _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, CAMERA_JPEG_QUALITY])
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n'


INDEX_HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>丛雨桌宠 + 手势联动</title>
  <style>
    html, body {
      margin: 0; padding: 0; width: 100%; height: 100%;
      background: transparent;
      color: #fff; font-family: sans-serif; overflow: hidden;
    }
    #app {
      position: fixed; inset: 0 420px 0 0;
    }
    .hud {
      position: fixed; left: 16px; top: 16px; z-index: 20;
      background: rgba(0,0,0,.50); padding: 12px 14px; border-radius: 12px;
      backdrop-filter: blur(6px); max-width: 520px;
    }
    .panel {
      position: fixed; top: 0; right: 0; width: 420px; height: 100vh;
      overflow-y: auto; box-sizing: border-box;
      background: rgba(9, 13, 20, 0.94); border-left: 1px solid rgba(255,255,255,.08);
      padding: 16px;
    }
    .bubble {
      position: fixed; left: 32px; bottom: 36px; z-index: 25;
      min-width: 120px; max-width: 420px;
      padding: 12px 16px; border-radius: 18px;
      background: rgba(255,255,255,.92); color: #222;
      box-shadow: 0 8px 24px rgba(0,0,0,.25);
      opacity: 0; transform: translateY(12px);
      transition: all .25s ease; pointer-events: none;
      font-size: 16px;
    }
    .bubble.show { opacity: 1; transform: translateY(0); }
    .hearts { position: fixed; inset: 0 420px 0 0; pointer-events: none; overflow: hidden; z-index: 15; }
    .heart {
      position: absolute; bottom: 80px; left: 50%;
      font-size: 26px; opacity: 0;
      animation: floatHeart 1.8s ease-out forwards;
      filter: drop-shadow(0 4px 10px rgba(255, 64, 128, 0.35));
    }
    @keyframes floatHeart {
      0% { transform: translate(0,0) scale(.7); opacity: 0; }
      10% { opacity: 1; }
      100% { transform: translate(var(--dx), -220px) scale(1.25); opacity: 0; }
    }
    .hud code, .panel code { color: #8ef; }
    .status { margin-top: 6px; color: #9f9; }
    .err { color: #ff8f8f; white-space: pre-wrap; font-size: 13px; }
    .dbg { color: #ffd580; white-space: pre-wrap; font-size: 12px; margin-top: 8px; }
    h2, h3 { margin: 10px 0 8px; }
    .section { margin-bottom: 18px; }
    .btns { display: flex; flex-wrap: wrap; gap: 8px; }
    button {
      padding: 8px 10px; border: 0; border-radius: 8px; background:#19c37d; color:white; cursor:pointer;
      font-size: 13px;
    }
    button.secondary { background: #3a78ff; }
    button.warn { background: #ff8c42; }
    .item {
      margin: 8px 0; padding: 8px; background: rgba(255,255,255,.05); border-radius: 8px;
      font-size: 12px; line-height: 1.4;
    }
    .muted { color: #c2c7d0; font-size: 12px; }
    .camera-box img { width: 100%; border-radius: 10px; border: 2px solid rgba(0,255,136,.45); }
    .gesture-now { font-size: 16px; color: #9f9; margin: 6px 0; }
    .mono { font-family: monospace; word-break: break-word; }
  </style>
  <script src="/runtime/pixi.js-legacy.min.js"></script>
  <script>
    window.globalThis = window;
    window.self = window;
  </script>
  <script src="/runtime/live2dcubismcore.min.js"></script>
  <script>
    window.Live2D = window.Live2D || {};
    window.Live2DMotion = window.Live2DMotion || function(){};
    window.AMotion = window.AMotion || function(){};
    window.PhysicsHair = window.PhysicsHair || function(){};
    window.MotionQueueManager = window.MotionQueueManager || function(){
      this.isFinished = function(){ return true; };
      this.startMotion = function(){ return null; };
      this.stopAllMotions = function(){};
      this.doUpdateMotion = function(){ return null; };
      this.setEventCallback = function(){};
      this.release = function(){};
    };
    window.Live2DModelWebGL = window.Live2DModelWebGL || { loadModel: function(){ return null; } };
    window.Live2D.CubismModel = window.Live2D.CubismModel || function() {};
    window.PIXI = window.PIXI || {};
    window.PIXI.live2d = window.PIXI.live2d || {};
    window.globalThis.PIXI = window.PIXI;
  </script>
  <script src="/runtime/cubism4.min.js"></script>
  <script src="/runtime/index.min.js"></script>
</head>
<body>
  <div class="hud">
    <div>丛雨桌宠 + 手势联动</div>
    <div>模型：<code>__MODEL_FILE__</code></div>
    <div id="status" class="status">页面已打开，等待脚本...</div>
    <div id="error" class="err"></div>
    <div id="debug" class="dbg"></div>
  </div>

  <div id="app"></div>
  <div id="hearts" class="hearts"></div>
  <div id="bubble" class="bubble"></div>

  <div class="panel">
    <h2>控制面板</h2>
    <div class="section camera-box">
      <h3>摄像头 / 手势</h3>
      <img src="/video_feed" alt="gesture camera" />
      <div class="gesture-now">当前手势：<span id="gestureNow">普通手势</span></div>
      <div class="muted mono">联动：<span id="gestureMapping">等待中...</span></div>
    </div>

    <div class="section">
      <div class="btns">
        <button class="secondary" onclick="resetModelView()">重置位置</button>
        <button class="secondary" onclick="playIdle()">播放 Idle</button>
        <button class="secondary" onclick="triggerWaveMode()">挥手</button>
        <button class="secondary" onclick="triggerHeartMode()">比心</button>
        <button class="secondary" onclick="playAllMotions()">轮播全部动作</button>
        <button class="secondary" onclick="playAllExpressions()">轮播全部表情</button>
      </div>
    </div>

    <div class="section">
      <h3>参数动作测试</h3>
      <div id="paramActions"></div>
    </div>

    <div class="section">
      <h3>Expressions</h3>
      <div id="expressions"></div>
    </div>

    <div class="section">
      <h3>Motions</h3>
      <div id="motions"></div>
    </div>
  </div>

  <script>
    const MODEL_URL = '/model/' + encodeURIComponent('__MODEL_FILE__');
    const EXPRESSIONS = __EXPRESSIONS_JSON__;
    const MOTION_GROUPS = __MOTIONS_JSON__;
    const GESTURE_MAP = __GESTURE_MAP_JSON__;
    const GESTURE_COOLDOWN = __GESTURE_COOLDOWN__;
    const PARAM_ACTIONS = [
      { key: 'ParamCheek42', label: '动作-正常' },
      { key: 'ParamCheek23', label: '动作-握手' },
      { key: 'ParamCheek40', label: '动作-伸手' },
      { key: 'ParamCheek43', label: '动作-比心' },
      { key: 'ParamCheek24', label: '动作-唱歌' },
      { key: 'ParamCheek21', label: '动作-心配' },
      { key: 'ParamCheek22', label: '动作-起爆器' },
      { key: 'ParamCheek5', label: '害羞' },
      { key: 'ParamCheek12', label: '喜-嘴巴' },
      { key: 'ParamCheek13', label: '怒-嘴巴' },
      { key: 'ParamCheek37', label: '嘟嘴' },
      { key: 'ParamCheek1', label: '高光' },
      { key: 'ParamCheek2', label: '星星眼' },
      { key: 'ParamCheek7', label: '泪珠' }
    ];

    let live2dModel = null;
    let app = null;
    let baseScale = 1;
    let gesturePollTimer = null;
    let gestureActionRunning = false;
    let gestureCooldownUntil = 0;
    let queuedGesture = null;
    let lastQueuedGestureAt = 0;

    function setStatus(text) { document.getElementById('status').textContent = text; }
    function setError(text) { document.getElementById('error').textContent = text || ''; }
    function setDebug(text) { document.getElementById('debug').textContent = text || ''; }

    function showBubble(text, ms = 1800) {
      const el = document.getElementById('bubble');
      el.textContent = text;
      el.classList.add('show');
      clearTimeout(showBubble._timer);
      showBubble._timer = setTimeout(() => el.classList.remove('show'), ms);
    }

    function burstHearts(count = 8) {
      const root = document.getElementById('hearts');
      for (let i = 0; i < count; i++) {
        const el = document.createElement('div');
        el.className = 'heart';
        el.textContent = ['❤','💗','💖','💕'][i % 4];
        el.style.left = (38 + Math.random() * 24) + '%';
        el.style.setProperty('--dx', ((Math.random() - 0.5) * 180).toFixed(0) + 'px');
        el.style.animationDelay = (Math.random() * 0.25).toFixed(2) + 's';
        root.appendChild(el);
        setTimeout(() => el.remove(), 2200);
      }
    }

    function resetModelView() {
      if (!live2dModel) return;
      placeModel();
      setStatus('已重置模型位置');
    }

    function isPetMode() {
      const p = new URLSearchParams(window.location.search);
      return p.get('mode') === 'pet';
    }

    function isSolidBgMode() {
      const p = new URLSearchParams(window.location.search);
      return p.get('bg') === 'solid';
    }

    function placeModel() {
      if (!live2dModel) return;
      const petMode = isPetMode();
      const sidebar = petMode ? 0 : 420;
      const availableWidth = window.innerWidth - sidebar;
      const availableHeight = window.innerHeight;
      const scale = Math.min(availableWidth / live2dModel.width, availableHeight / live2dModel.height) * (petMode ? 1.08 : 0.84);
      baseScale = scale;
      live2dModel.scale.set(scale);
      if (live2dModel.anchor) live2dModel.anchor.set(0.5, 1.0);
      live2dModel.x = petMode ? window.innerWidth * 0.50 : availableWidth / 2;
      live2dModel.y = petMode ? window.innerHeight * 0.92 : window.innerHeight * 0.96;
      try { live2dModel.visible = true; live2dModel.renderable = true; live2dModel.alpha = 1; } catch (e) {}
    }

    async function playExpression(index) {
      if (!live2dModel) return;
      try {
        if (live2dModel.expression) {
          await live2dModel.expression(index);
          setStatus('已切换表情 #' + index);
        }
      } catch (err) {
        setError(String(err && err.stack ? err.stack : err));
      }
    }

    async function playMotion(group, index) {
      if (!live2dModel) return;
      try {
        if (live2dModel.motion) {
          await live2dModel.motion(group, index);
          setStatus(`已播放动作 ${group}[${index}]`);
        }
      } catch (err) {
        setError(String(err && err.stack ? err.stack : err));
      }
    }

    async function playIdle() { return playMotion('Idle', 0); }

    async function playAllExpressions() {
      if (!EXPRESSIONS.length) return;
      setStatus('开始轮播全部表情...');
      for (let i = 0; i < EXPRESSIONS.length; i++) {
        await playExpression(i);
        await new Promise(r => setTimeout(r, 1200));
      }
      setStatus('表情轮播完成');
    }

    async function playAllMotions() {
      setStatus('开始轮播全部动作...');
      for (const group of MOTION_GROUPS) {
        for (const item of group.items) {
          await playMotion(group.group, item.index);
          await new Promise(r => setTimeout(r, 2600));
        }
      }
      setStatus('动作轮播完成');
    }

    function findMotionCandidates() {
      const flat = [];
      for (const group of MOTION_GROUPS) {
        for (const item of group.items) {
          flat.push({ group: group.group, index: item.index, file: item.file || '', text: item.text || '' });
        }
      }
      const preferWave = flat.find(x => /待机|wave|hello|greet|摸头|唱歌/i.test(x.file + ' ' + x.text + ' ' + x.group)) || flat[0] || null;
      const preferHeart = flat.find(x => /啵|love|心|开心|盯|0721/i.test(x.file + ' ' + x.text + ' ' + x.group)) || flat[Math.min(1, flat.length - 1)] || null;
      return { preferWave, preferHeart };
    }

    async function triggerWaveMode() {
      const picked = findMotionCandidates().preferWave;
      showBubble('你好呀~');
      setStatus('触发：挥手模式');
      if (picked) await playMotion(picked.group, picked.index);
    }

    async function triggerHeartMode() {
      const picked = findMotionCandidates().preferHeart;
      showBubble('给你一个小心心~', 2200);
      burstHearts(10);
      setStatus('触发：比心模式');
      if (EXPRESSIONS.length) {
        try { await playExpression(Math.min(1, EXPRESSIONS.length - 1)); } catch (e) {}
      }
      if (picked) await playMotion(picked.group, picked.index);
    }

    function resetParamActions() {
      if (!live2dModel || !live2dModel.internalModel || !live2dModel.internalModel.coreModel) return;
      const core = live2dModel.internalModel.coreModel;
      for (const item of PARAM_ACTIONS) {
        try { core.setParameterValueById(item.key, 0); } catch (e) {}
      }
    }

    async function setParamAction(key, value = 1) {
      if (!live2dModel || !live2dModel.internalModel || !live2dModel.internalModel.coreModel) return;
      try {
        const core = live2dModel.internalModel.coreModel;
        resetParamActions();
        core.setParameterValueById(key, value);
        setStatus('已设置参数动作: ' + key + '=' + value);
      } catch (err) {
        setError(String(err && err.stack ? err.stack : err));
      }
    }

    async function applyGestureAction(gesture, mapping, silent=false) {
      if (!mapping) return;
      gestureActionRunning = true;
      try {
        if (mapping.motion_group) {
          await playMotion(mapping.motion_group, mapping.motion_index || 0);
        }
        if (mapping.param) {
          await setParamAction(mapping.param, 1);
        }
        if (typeof mapping.expression === 'number') {
          try { await playExpression(mapping.expression); } catch (e) {}
        }
        if (gesture === '比心 ❤️') {
          burstHearts(10);
          showBubble('给你一个小心心~', 2200);
        } else if (gesture === '挥手动作 👋') {
          showBubble('这个现在走摇头动作~');
        } else if (gesture === '大拇指动作 👍') {
          showBubble('大拇指对应 motion09 ~');
        } else if (gesture === 'OK手势 ✔️') {
          showBubble('OK 对应 Tapxiongbu');
        } else if (gesture === 'yes手势 👌') {
          showBubble('✌ 对应 Taphair');
        } else if (gesture === '握拳动作 👊') {
          showBubble('认真模式！');
        } else if (gesture === '挥拳动作 ✊') {
          showBubble('挥拳对应 motion10！');
        }
        if (!silent) {
          const parts = [];
          if (mapping.motion_group) parts.push(mapping.motion_group + '[' + (mapping.motion_index || 0) + ']');
          if (mapping.param) parts.push(mapping.param);
          if (typeof mapping.expression === 'number') parts.push('exp' + (mapping.expression + 1));
          setStatus('手势联动：' + gesture + ' -> ' + parts.join(' + ') + '，进入30秒冷却');
        }
      } finally {
        gestureCooldownUntil = Date.now() + (GESTURE_COOLDOWN * 1000);
        gestureActionRunning = false;
      }
    }

    async function maybeRunQueuedGesture() {
      if (gestureActionRunning) return;
      if (!queuedGesture || !queuedGesture.mapping) return;
      if (Date.now() < gestureCooldownUntil) return;
      const job = queuedGesture;
      queuedGesture = null;
      await applyGestureAction(job.gesture, job.mapping);
    }

    async function pollGestureState() {
      try {
        const r = await fetch('/api/gesture_state');
        const j = await r.json();
        document.getElementById('gestureNow').textContent = j.current_gesture || '普通手势';
        const mapping = j.mapping;
        document.getElementById('gestureMapping').textContent = mapping ? `${mapping.param}${typeof mapping.expression === 'number' ? (' + exp' + (mapping.expression + 1)) : ''} / ${mapping.label || ''}` : '无映射';

        if (mapping && j.current_gesture && j.current_gesture !== '普通手势') {
          const isSameQueued = queuedGesture && queuedGesture.gesture === j.current_gesture;
          if (!isSameQueued) {
            queuedGesture = { gesture: j.current_gesture, mapping: mapping };
            lastQueuedGestureAt = Date.now();
          }
        }

        if (gestureActionRunning) {
          const left = Math.max(0, Math.ceil((gestureCooldownUntil - Date.now()) / 1000));
          if (left > 0) setStatus('动作执行中，冷却剩余 ' + left + ' 秒');
          return;
        }

        if (Date.now() < gestureCooldownUntil) {
          const left = Math.max(0, Math.ceil((gestureCooldownUntil - Date.now()) / 1000));
          setStatus('冷却中，剩余 ' + left + ' 秒');
          return;
        }

        await maybeRunQueuedGesture();
      } catch (e) {
        setDebug('poll gesture failed: ' + e);
      }
    }

    function renderControls() {
      const expRoot = document.getElementById('expressions');
      const motionRoot = document.getElementById('motions');
      const paramRoot = document.getElementById('paramActions');
      expRoot.innerHTML = '';
      motionRoot.innerHTML = '';
      paramRoot.innerHTML = '';

      PARAM_ACTIONS.forEach((item) => {
        const div = document.createElement('div');
        div.className = 'item';
        div.innerHTML = `
          <div><b><code>${item.key}</code></b></div>
          <div class="muted">${item.label}</div>
          <div class="btns">
            <button onclick="setParamAction('${item.key}', 1)">设为 1</button>
            <button class="secondary" onclick="setParamAction('${item.key}', 0.5)">设为 0.5</button>
            <button class="warn" onclick="setParamAction('${item.key}', 0)">清零</button>
          </div>
        `;
        paramRoot.appendChild(div);
      });

      if (!EXPRESSIONS.length) {
        expRoot.innerHTML = '<div class="muted">无 expressions</div>';
      } else {
        EXPRESSIONS.forEach((exp, idx) => {
          const div = document.createElement('div');
          div.className = 'item';
          div.innerHTML = `
            <div><b>#${idx}</b> <code>${exp.Name || 'unnamed'}</code></div>
            <div class="muted">${exp.File || ''}</div>
            <div class="btns"><button onclick="playExpression(${idx})">播放表情</button></div>
          `;
          expRoot.appendChild(div);
        });
      }

      if (!MOTION_GROUPS.length) {
        motionRoot.innerHTML = '<div class="muted">无 motions</div>';
      } else {
        MOTION_GROUPS.forEach((group) => {
          const wrap = document.createElement('div');
          wrap.className = 'item';
          let html = `<div><b>${group.group}</b></div>`;
          group.items.forEach((item) => {
            html += `
              <div style="margin-top:8px; padding-top:8px; border-top:1px solid rgba(255,255,255,.08)">
                <div><b>[${item.index}]</b> <code>${item.file}</code></div>
                ${item.sound ? `<div class="muted">声音: ${item.sound}</div>` : ''}
                ${item.text ? `<div class="muted">文本: ${item.text}</div>` : ''}
                <div class="btns"><button class="warn" onclick="playMotion('${group.group}', ${item.index})">播放动作</button></div>
              </div>
            `;
          });
          wrap.innerHTML = html;
          motionRoot.appendChild(wrap);
        });
      }
    }

    window.addEventListener('load', async () => {
      try {
        const petMode = isPetMode();
        const solidBg = isSolidBgMode();
        document.body.style.background = solidBg ? '#223' : 'transparent';
        const hud = document.querySelector('.hud');
        const panel = document.querySelector('.panel');
        const appEl = document.getElementById('app');
        const hearts = document.getElementById('hearts');
        if (petMode) {
          if (hud) hud.style.display = 'none';
          if (panel) panel.style.display = 'none';
          if (appEl) appEl.style.inset = '0';
          if (hearts) hearts.style.inset = '0';
        }

        setDebug('loading model...');
        if (!(window.PIXI && window.PIXI.live2d && window.PIXI.live2d.Live2DModel)) {
          throw new Error('Live2DModel 不可用，请检查 runtime 加载');
        }

        app = new PIXI.Application({
          view: document.createElement('canvas'),
          autoStart: true,
          resizeTo: document.getElementById('app'),
          backgroundAlpha: solidBg ? 1 : 0,
          backgroundColor: solidBg ? 0x223344 : 0x000000,
          antialias: true,
        });
        document.getElementById('app').appendChild(app.view);

        live2dModel = await PIXI.live2d.Live2DModel.from(MODEL_URL);
        app.stage.addChild(live2dModel);
        placeModel();
        renderControls();
        live2dModel.interactive = false;

        try {
          const im = live2dModel.internalModel;
          if (im && im.focusController) {
            if ('x' in im.focusController) im.focusController.x = 0;
            if ('y' in im.focusController) im.focusController.y = 0;
            if ('targetX' in im.focusController) im.focusController.targetX = 0;
            if ('targetY' in im.focusController) im.focusController.targetY = 0;
          }
        } catch (e) {}

        app.ticker.add(() => {
          try {
            const im = live2dModel && live2dModel.internalModel;
            if (im && im.focusController) {
              if ('x' in im.focusController) im.focusController.x *= 0.2;
              if ('y' in im.focusController) im.focusController.y *= 0.2;
              if ('targetX' in im.focusController) im.focusController.targetX *= 0.2;
              if ('targetY' in im.focusController) im.focusController.targetY *= 0.2;
            }
          } catch (e) {}
          try {
            if (!live2dModel) return;
            const sx = live2dModel.scale.x;
            const sy = live2dModel.scale.y;
            const target = baseScale;
            const clampedX = Math.max(target * 0.92, Math.min(target * 1.02, sx));
            if (Math.abs(clampedX - sx) > 0.0001) live2dModel.scale.x = clampedX;
            if (Math.abs(sy - target) > target * 0.10) live2dModel.scale.y = target;
          } catch (e) {}
        });

        window.addEventListener('resize', placeModel);
        setStatus('模型加载成功，手势联动已启用 ✅');
        gesturePollTimer = setInterval(pollGestureState, 350);
        pollGestureState();
      } catch (err) {
        console.error(err);
        setStatus('加载失败');
        setError(String(err && err.stack ? err.stack : err));
      }
    });
  </script>
</body>
</html>
'''

INDEX_HTML = INDEX_HTML.replace('__MODEL_FILE__', MODEL_FILE)
INDEX_HTML = INDEX_HTML.replace('__EXPRESSIONS_JSON__', json.dumps(EXPRESSIONS, ensure_ascii=False))
INDEX_HTML = INDEX_HTML.replace('__MOTIONS_JSON__', json.dumps(MOTION_GROUPS, ensure_ascii=False))
INDEX_HTML = INDEX_HTML.replace('__GESTURE_MAP_JSON__', json.dumps(GESTURE_MAP, ensure_ascii=False))
INDEX_HTML = INDEX_HTML.replace('__GESTURE_COOLDOWN__', json.dumps(GESTURE_COOLDOWN, ensure_ascii=False))

CONTROL_HTML = """<!doctype html>
<html lang=\"zh-CN\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>丛雨合并控制台</title>
  <style>
    body { margin:0; padding:20px; background:#171b22; color:#fff; font-family:sans-serif; }
    h1,h2 { margin:0 0 12px; }
    .card { background:#232833; border-radius:14px; padding:16px; margin-bottom:16px; }
    .btns { display:flex; flex-wrap:wrap; gap:10px; }
    button { padding:10px 14px; border:0; border-radius:10px; background:#3a78ff; color:#fff; cursor:pointer; }
    .muted { color:#aeb7c4; font-size:13px; margin-top:8px; }
    .mono { font-family:monospace; word-break:break-word; }
  </style>
</head>
<body>
  <h1>丛雨桌宠 + 手势合并控制台</h1>
  <div class=\"card\">
    <h2>页面入口</h2>
    <div class=\"btns\">
      <button onclick=\"window.open('/?mode=pet', '_blank')\">打开桌宠</button>
      <button onclick=\"window.open('/?mode=debug', '_blank')\">打开调试页</button>
      <button onclick=\"window.open('/debug-actions', '_blank')\">打开动作总览页</button>
    </div>
    <div class=\"muted\">一个进程同时跑 Live2D 和 MediaPipe 手势识别。</div>
  </div>
  <div class=\"card\">
    <h2>手势映射</h2>
    <div class=\"mono\">__GESTURE_MAP_PRETTY__</div>
  </div>
</body>
</html>
""".replace('__GESTURE_MAP_PRETTY__', json.dumps(GESTURE_MAP, ensure_ascii=False, indent=2))

DEBUG_ACTIONS_HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>丛雨动作总览调试页</title>
  <style>
    body { margin:0; background:#10141c; color:#fff; font-family:sans-serif; }
    .wrap { display:grid; grid-template-columns: 420px 1fr; min-height:100vh; }
    .side { padding:16px; background:rgba(14,18,26,.96); border-right:1px solid rgba(255,255,255,.08); overflow-y:auto; }
    .main { position:relative; min-height:100vh; }
    #app { position:absolute; inset:0; }
    .section { margin-bottom:18px; background:rgba(255,255,255,.04); border-radius:12px; padding:12px; }
    h1,h2,h3 { margin:0 0 10px; }
    .btns { display:flex; flex-wrap:wrap; gap:8px; }
    button { padding:8px 10px; border:0; border-radius:8px; background:#3a78ff; color:white; cursor:pointer; font-size:13px; }
    button.good { background:#19c37d; }
    button.warn { background:#ff8c42; }
    .item { margin:8px 0; padding:8px; background:rgba(255,255,255,.05); border-radius:8px; font-size:12px; line-height:1.4; }
    .muted { color:#bfc7d5; font-size:12px; }
    .mono { font-family:monospace; word-break:break-word; }
    .camera { width:100%; border-radius:10px; border:2px solid rgba(0,255,136,.45); }
    .status { color:#9f9; margin-top:8px; white-space:pre-wrap; }
    .bubble {
      position: fixed; left: 460px; bottom: 36px; z-index: 25;
      min-width: 120px; max-width: 420px;
      padding: 12px 16px; border-radius: 18px;
      background: rgba(255,255,255,.92); color: #222;
      box-shadow: 0 8px 24px rgba(0,0,0,.25);
      opacity: 0; transform: translateY(12px);
      transition: all .25s ease; pointer-events: none; font-size:16px;
    }
    .bubble.show { opacity:1; transform:translateY(0); }
    .hearts { position: fixed; inset: 0 0 0 420px; pointer-events:none; overflow:hidden; z-index:15; }
    .heart {
      position:absolute; bottom:80px; left:50%; font-size:26px; opacity:0;
      animation: floatHeart 1.8s ease-out forwards;
      filter: drop-shadow(0 4px 10px rgba(255, 64, 128, 0.35));
    }
    @keyframes floatHeart {
      0% { transform: translate(0,0) scale(.7); opacity:0; }
      10% { opacity:1; }
      100% { transform: translate(var(--dx), -220px) scale(1.25); opacity:0; }
    }
  </style>
  <script src="/runtime/pixi.js-legacy.min.js"></script>
  <script>
    window.globalThis = window;
    window.self = window;
  </script>
  <script src="/runtime/live2dcubismcore.min.js"></script>
  <script>
    window.Live2D = window.Live2D || {};
    window.Live2DMotion = window.Live2DMotion || function(){};
    window.AMotion = window.AMotion || function(){};
    window.PhysicsHair = window.PhysicsHair || function(){};
    window.MotionQueueManager = window.MotionQueueManager || function(){
      this.isFinished = function(){ return true; };
      this.startMotion = function(){ return null; };
      this.stopAllMotions = function(){};
      this.doUpdateMotion = function(){ return null; };
      this.setEventCallback = function(){};
      this.release = function(){};
    };
    window.Live2DModelWebGL = window.Live2DModelWebGL || { loadModel: function(){ return null; } };
    window.Live2D.CubismModel = window.Live2D.CubismModel || function() {};
    window.PIXI = window.PIXI || {};
    window.PIXI.live2d = window.PIXI.live2d || {};
    window.globalThis.PIXI = window.PIXI;
  </script>
  <script src="/runtime/cubism4.min.js"></script>
  <script src="/runtime/index.min.js"></script>
</head>
<body>
  <div class="wrap">
    <div class="side">
      <div class="section">
        <h1>动作总览调试页</h1>
        <div class="muted">这里专门用来人工点动作、表情、motion，对照摄像头手势。</div>
      </div>

      <div class="section">
        <h2>摄像头 / 当前手势</h2>
        <img class="camera" src="/video_feed" alt="camera" />
        <div class="status" id="gestureStatus">手势状态加载中...</div>
      </div>

      <div class="section">
        <h2>快捷操作</h2>
        <div class="btns">
          <button class="good" onclick="resetModelView()">重置位置</button>
          <button onclick="playIdle()">播放 Idle</button>
          <button onclick="resetParamActions()">清空参数动作</button>
          <button onclick="playAllExpressions()">轮播表情</button>
          <button onclick="playAllMotions()">轮播 motions</button>
        </div>
        <div class="status" id="runStatus">等待操作...</div>
      </div>

      <div class="section">
        <h2>参数动作</h2>
        <div id="paramActions"></div>
      </div>

      <div class="section">
        <h2>Expressions</h2>
        <div id="expressions"></div>
      </div>

      <div class="section">
        <h2>Motions</h2>
        <div id="motions"></div>
      </div>
    </div>
    <div class="main">
      <div id="app"></div>
    </div>
  </div>
  <div id="hearts" class="hearts"></div>
  <div id="bubble" class="bubble"></div>

  <script>
    const MODEL_URL = '/model/' + encodeURIComponent('__MODEL_FILE__');
    const EXPRESSIONS = __EXPRESSIONS_JSON__;
    const MOTION_GROUPS = __MOTIONS_JSON__;
    const PARAM_ACTIONS = [
      { key: 'ParamCheek42', label: '动作-正常' },
      { key: 'ParamCheek23', label: '动作-握手' },
      { key: 'ParamCheek40', label: '动作-伸手' },
      { key: 'ParamCheek43', label: '动作-比心' },
      { key: 'ParamCheek24', label: '动作-唱歌' },
      { key: 'ParamCheek21', label: '动作-心配' },
      { key: 'ParamCheek22', label: '动作-起爆器' },
      { key: 'ParamCheek5', label: '害羞' },
      { key: 'ParamCheek12', label: '喜-嘴巴' },
      { key: 'ParamCheek13', label: '怒-嘴巴' },
      { key: 'ParamCheek37', label: '嘟嘴' },
      { key: 'ParamCheek1', label: '高光' },
      { key: 'ParamCheek2', label: '星星眼' },
      { key: 'ParamCheek7', label: '泪珠' }
    ];

    let live2dModel = null;
    let app = null;
    let baseScale = 1;

    function setRunStatus(text) {
      document.getElementById('runStatus').textContent = text;
    }

    function showBubble(text, ms = 1800) {
      const el = document.getElementById('bubble');
      el.textContent = text;
      el.classList.add('show');
      clearTimeout(showBubble._timer);
      showBubble._timer = setTimeout(() => el.classList.remove('show'), ms);
    }

    function burstHearts(count = 8) {
      const root = document.getElementById('hearts');
      for (let i = 0; i < count; i++) {
        const el = document.createElement('div');
        el.className = 'heart';
        el.textContent = ['❤','💗','💖','💕'][i % 4];
        el.style.left = (38 + Math.random() * 24) + '%';
        el.style.setProperty('--dx', ((Math.random() - 0.5) * 180).toFixed(0) + 'px');
        el.style.animationDelay = (Math.random() * 0.25).toFixed(2) + 's';
        root.appendChild(el);
        setTimeout(() => el.remove(), 2200);
      }
    }

    function placeModel() {
      if (!live2dModel) return;
      const availableWidth = window.innerWidth - 420;
      const availableHeight = window.innerHeight;
      const scale = Math.min(availableWidth / live2dModel.width, availableHeight / live2dModel.height) * 0.88;
      baseScale = scale;
      live2dModel.scale.set(scale);
      if (live2dModel.anchor) live2dModel.anchor.set(0.5, 1.0);
      live2dModel.x = 420 + availableWidth / 2;
      live2dModel.y = window.innerHeight * 0.96;
      try { live2dModel.visible = true; live2dModel.renderable = true; live2dModel.alpha = 1; } catch (e) {}
    }

    function resetModelView() {
      placeModel();
      setRunStatus('已重置模型位置');
    }

    async function playExpression(index) {
      if (!live2dModel || !live2dModel.expression) return;
      await live2dModel.expression(index);
      setRunStatus('已切换表情 #' + index);
    }

    async function playMotion(group, index) {
      if (!live2dModel || !live2dModel.motion) return;
      await live2dModel.motion(group, index);
      setRunStatus(`已播放动作 ${group}[${index}]`);
    }

    async function playIdle() { return playMotion('Idle', 0); }

    async function playAllExpressions() {
      setRunStatus('开始轮播全部表情...');
      for (let i = 0; i < EXPRESSIONS.length; i++) {
        await playExpression(i);
        await new Promise(r => setTimeout(r, 1200));
      }
      setRunStatus('表情轮播完成');
    }

    async function playAllMotions() {
      setRunStatus('开始轮播全部 motions...');
      for (const group of MOTION_GROUPS) {
        for (const item of group.items) {
          await playMotion(group.group, item.index);
          await new Promise(r => setTimeout(r, 2600));
        }
      }
      setRunStatus('motions 轮播完成');
    }

    function resetParamActions() {
      if (!live2dModel || !live2dModel.internalModel || !live2dModel.internalModel.coreModel) return;
      const core = live2dModel.internalModel.coreModel;
      for (const item of PARAM_ACTIONS) {
        try { core.setParameterValueById(item.key, 0); } catch (e) {}
      }
      setRunStatus('已清空参数动作');
    }

    async function setParamAction(key, value = 1) {
      if (!live2dModel || !live2dModel.internalModel || !live2dModel.internalModel.coreModel) return;
      const core = live2dModel.internalModel.coreModel;
      resetParamActions();
      core.setParameterValueById(key, value);
      setRunStatus('已设置参数动作: ' + key + '=' + value);
      if (key === 'ParamCheek43') {
        burstHearts(10);
        showBubble('这个像比心吗？', 2200);
      } else if (key === 'ParamCheek23') {
        showBubble('这个像挥手吗？');
      }
    }

    function renderControls() {
      const expRoot = document.getElementById('expressions');
      const motionRoot = document.getElementById('motions');
      const paramRoot = document.getElementById('paramActions');
      expRoot.innerHTML = '';
      motionRoot.innerHTML = '';
      paramRoot.innerHTML = '';

      PARAM_ACTIONS.forEach((item) => {
        const div = document.createElement('div');
        div.className = 'item';
        div.innerHTML = `
          <div><b><code>${item.key}</code></b></div>
          <div class="muted">${item.label}</div>
          <div class="btns">
            <button onclick="setParamAction('${item.key}', 1)">设为 1</button>
            <button onclick="setParamAction('${item.key}', 0.5)">设为 0.5</button>
            <button class="warn" onclick="setParamAction('${item.key}', 0)">清零</button>
          </div>
        `;
        paramRoot.appendChild(div);
      });

      EXPRESSIONS.forEach((exp, idx) => {
        const div = document.createElement('div');
        div.className = 'item';
        div.innerHTML = `
          <div><b>#${idx}</b> <code>${exp.Name || 'unnamed'}</code></div>
          <div class="muted">${exp.File || ''}</div>
          <div class="btns"><button onclick="playExpression(${idx})">播放表情</button></div>
        `;
        expRoot.appendChild(div);
      });

      MOTION_GROUPS.forEach((group) => {
        const wrap = document.createElement('div');
        wrap.className = 'item';
        let html = `<div><b>${group.group}</b></div>`;
        group.items.forEach((item) => {
          html += `
            <div style="margin-top:8px; padding-top:8px; border-top:1px solid rgba(255,255,255,.08)">
              <div><b>[${item.index}]</b> <code>${item.file}</code></div>
              ${item.sound ? `<div class="muted">声音: ${item.sound}</div>` : ''}
              ${item.text ? `<div class="muted">文本: ${item.text}</div>` : ''}
              <div class="btns"><button class="warn" onclick="playMotion('${group.group}', ${item.index})">播放 motion</button></div>
            </div>
          `;
        });
        wrap.innerHTML = html;
        motionRoot.appendChild(wrap);
      });
    }

    async function pollGestureState() {
      try {
        const r = await fetch('/api/gesture_state');
        const j = await r.json();
        document.getElementById('gestureStatus').textContent = '当前手势: ' + (j.current_gesture || '普通手势');
      } catch (e) {
        document.getElementById('gestureStatus').textContent = '读取手势失败: ' + e;
      }
    }

    window.addEventListener('load', async () => {
      app = new PIXI.Application({
        view: document.createElement('canvas'),
        autoStart: true,
        resizeTo: document.querySelector('.main'),
        backgroundAlpha: 0,
        antialias: true,
      });
      document.getElementById('app').appendChild(app.view);
      live2dModel = await PIXI.live2d.Live2DModel.from(MODEL_URL);
      app.stage.addChild(live2dModel);
      placeModel();
      renderControls();
      live2dModel.interactive = false;

      app.ticker.add(() => {
        try {
          const im = live2dModel && live2dModel.internalModel;
          if (im && im.focusController) {
            if ('x' in im.focusController) im.focusController.x *= 0.2;
            if ('y' in im.focusController) im.focusController.y *= 0.2;
            if ('targetX' in im.focusController) im.focusController.targetX *= 0.2;
            if ('targetY' in im.focusController) im.focusController.targetY *= 0.2;
          }
        } catch (e) {}
      });

      window.addEventListener('resize', placeModel);
      setRunStatus('模型加载成功，可以开始点动作了 ✅');
      setInterval(pollGestureState, 500);
      pollGestureState();
    });
  </script>
</body>
</html>
'''

DEBUG_ACTIONS_HTML = DEBUG_ACTIONS_HTML.replace('__MODEL_FILE__', MODEL_FILE)
DEBUG_ACTIONS_HTML = DEBUG_ACTIONS_HTML.replace('__EXPRESSIONS_JSON__', json.dumps(EXPRESSIONS, ensure_ascii=False))
DEBUG_ACTIONS_HTML = DEBUG_ACTIONS_HTML.replace('__MOTIONS_JSON__', json.dumps(MOTION_GROUPS, ensure_ascii=False))


@app.route('/')
def index():
    return Response(INDEX_HTML, mimetype='text/html; charset=utf-8')


@app.route('/control')
def control_page():
    return Response(CONTROL_HTML, mimetype='text/html; charset=utf-8')


@app.route('/debug-actions')
def debug_actions_page():
    return Response(DEBUG_ACTIONS_HTML, mimetype='text/html; charset=utf-8')


@app.route('/model/<path:filename>')
def model_files(filename):
    return send_from_directory(MODEL_DIR, filename)


@app.route('/runtime/<path:filename>')
def runtime_files(filename):
    return send_from_directory(RUNTIME_DIR, filename)


@app.route('/video_feed')
def feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/model_info')
def model_info():
    return jsonify({
        'model_file': MODEL_FILE,
        'expressions': EXPRESSIONS,
        'motions': MOTION_GROUPS,
        'gesture_map': GESTURE_MAP,
    })


@app.route('/api/gesture_state')
def gesture_state():
    return jsonify({
        'current_gesture': current_gesture,
        'mapping': GESTURE_MAP.get(current_gesture),
        'changed_at': last_gesture_change_at,
        'cooldown_seconds': GESTURE_COOLDOWN,
    })


@app.route('/__telemetry', methods=['POST'])
def page_telemetry():
    try:
        data = request.get_json(silent=True) or {}
    except Exception:
        data = {}
    PAGE_TELEMETRY['events'].append(data)
    PAGE_TELEMETRY['events'] = PAGE_TELEMETRY['events'][-50:]
    print('[PAGE_TELEMETRY]', json.dumps(data, ensure_ascii=False), flush=True)
    return jsonify({"ok": True, "count": len(PAGE_TELEMETRY['events'])})


@app.route('/__telemetry', methods=['GET'])
def page_telemetry_get():
    return jsonify(PAGE_TELEMETRY)


@app.route('/api/camera/start', methods=['POST'])
def camera_start():
    """手动启动摄像头和手势检测"""
    global camera, stop_thread, camera_thread_started, current_gesture
    if camera is not None and camera.isOpened():
        return jsonify({'ok': True, 'msg': '摄像头已运行', 'status': 'already_running'})
    ok = init_camera_device()
    if ok:
        ensure_camera_thread()
        current_gesture = '普通手势'
        return jsonify({'ok': True, 'msg': '摄像头启动成功', 'status': 'started'})
    return jsonify({'ok': False, 'msg': '摄像头启动失败，检查设备连接', 'status': 'failed'})


@app.route('/api/camera/stop', methods=['POST'])
def camera_stop():
    """手动停止摄像头"""
    global camera, stop_thread, camera_thread_started, current_gesture
    stop_thread = True
    if camera is not None and camera.isOpened():
        camera.release()
    camera = None
    camera_thread_started = False
    current_gesture = '普通手势'
    stop_thread = False
    return jsonify({'ok': True, 'msg': '摄像头已停止', 'status': 'stopped'})


@app.route('/api/camera/status', methods=['GET'])
def camera_api_status():
    """获取摄像头状态"""
    cam_ok = camera is not None and camera.isOpened()
    streaming = camera_thread_started
    return jsonify({
        'camera_on': cam_ok,
        'streaming': streaming,
        'current_gesture': current_gesture,
    })


if __name__ == '__main__':
    print('=' * 60)
    print(' 丛雨桌宠 + 手势识别 合并版 ')
    print('=' * 60)
    print(f'模型目录: {MODEL_DIR}')
    print(f'运行库目录: {RUNTIME_DIR}')
    print(f'入口文件: {MODEL_FILE}')
    init_mediapipe()
    # 如果不是无摄像头模式，启动时初始化摄像头
    disable_camera = os.environ.get('DISABLE_CAMERA_ON_START', '0') == '1'
    if not disable_camera:
        init_camera_device()
        ensure_camera_thread()
        print('📷 摄像头已初始化')
    else:
        print('📷 摄像头延迟初始化模式（可通过 API 或页面按钮启动）')
    print(f'访问地址: http://{HOST_IP}:{WEB_PORT}')
    print(f'控制台地址: http://{HOST_IP}:{WEB_PORT}/control')
    try:
        app.run(host='0.0.0.0', port=WEB_PORT, debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        stop_thread = True
    finally:
        if camera is not None and camera.isOpened():
            camera.release()
        if hands is not None:
            hands.close()
        print('✅ 已退出')
