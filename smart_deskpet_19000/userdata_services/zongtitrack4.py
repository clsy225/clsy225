import os
import cv2
import time
import threading
import queue
import numpy as np
import requests
import base64
import re
import subprocess
import glob
from flask import Flask, Response, request, jsonify
from collections import deque
import warnings
warnings.filterwarnings("ignore")

# ===================== 核心配置 =====================
RKNN_MODEL_PATH = "/home/linaro/yolo11n.rknn"
WEB_PORT = 8088
OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = (640, 640)
TRACKER_CONFIDENCE_THRESH = 0.8
DETECT_INTERVAL = 0.05

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

# ===================== 通义千问VL配置 =====================
QIANWEN_API_KEY = os.environ.get("QIANWEN_API_KEY", "")
QIANWEN_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
QIANWEN_MODEL = "qwen-vl-plus"

# ===================== TTS配置（自动识别杰理声卡）=====================
WINDOWS_TTS_IP = "192.168.137.1"
WINDOWS_TTS_PORT = 9880
LOCAL_TTS_URL = "http://127.0.0.1:9880/"

def test_tts_connectivity(url):
    try:
        response = requests.get(url, timeout=3)
        print(f"✅ TTS服务连通性测试成功：{url}")
        return True
    except Exception as e:
        print(f"❌ TTS服务连通性测试失败：{url}，错误：{str(e)}")
        return False

print("\n🔍 开始测试TTS服务连通性...")
WINDOWS_TTS_URL = f"http://{WINDOWS_TTS_IP}:{WINDOWS_TTS_PORT}/"
if test_tts_connectivity(WINDOWS_TTS_URL):
    TTS_API_URL = WINDOWS_TTS_URL
    print("✅ 优先使用Windows电脑的TTS服务")
elif test_tts_connectivity(LOCAL_TTS_URL):
    TTS_API_URL = LOCAL_TTS_URL
    print("⚠️  Windows TTS不通，使用开发板本地TTS服务")
else:
    TTS_API_URL = None
    print("❌ 所有TTS服务都无法访问，语音播报功能将不可用")

# ===================== 自动识别杰理UACDemoV1.0声卡 =====================
PEOPLE_LIMIT = 5
AUTO_ANALYZE_INTERVAL = 300
TEMP_DIR = "/home/linaro/vision_temp"
os.makedirs(TEMP_DIR, exist_ok=True)
CAPTURE_PATH = os.path.join(TEMP_DIR, "capture.jpg")

def get_usb_audio_card():
    try:
        result = subprocess.check_output("aplay -l", shell=True, text=True)
        for line in result.split("\n"):
            if "UACDemoV1.0" in line:
                card_num = line.split("card ")[1].split(":")[0]
                return card_num
    except Exception as e:
        print(f"⚠️  声卡识别异常：{e}")
    return None

AUDIO_CARD = get_usb_audio_card()
if AUDIO_CARD:
    PLAY_DEVICE = f"plughw:{AUDIO_CARD},0"
    print(f"✅ 自动匹配杰理USB音频设备：card{AUDIO_CARD} (UACDemoV1.0)")
else:
    PLAY_DEVICE = "default"
    print("⚠️  未找到杰理UACDemoV1.0设备，使用系统默认声卡")

# COCO 类别
CLASSES = ("person", "bicycle", "car","motorbike","aeroplane","bus","train","truck","boat","traffic light",
           "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant",
           "bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite",
           "baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife",
           "spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa",
           "pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard","cell phone","microwave",
           "oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush")

# ===================== 全局变量（核心修复：新增全局帧缓存）=====================
app = Flask(__name__)
camera = None
frame_queue = queue.Queue(maxsize=2)
stop_thread = False
fps_queue = deque(maxlen=30)
rknn_lite = None
last_analyze_time = 0

# ✅ 核心修复：新增全局最新帧缓存+锁，解决手动分析抢帧问题
latest_frame = None
frame_lock = threading.Lock()

# 状态管理
current_state = "detect"
object_id_counter = 0
current_detections = {}
track_info = {
    "target_id": None,
    "target_bbox": None,
    "tracker": None
}
state_lock = threading.Lock()

# ===================== TTS语音播报 =====================
def text_to_speech_play(text):
    if TTS_API_URL is None:
        print("❌ 无可用的TTS服务，跳过语音播报")
        return False

    tts_file = os.path.join(TEMP_DIR, "tts.wav")
    if os.path.exists(tts_file):
        try:
            os.remove(tts_file)
        except:
            pass
    try:
        print(f"🔊 合成语音：{text[:30]}...")
        response = requests.post(
            TTS_API_URL,
            json={
                "text": text,
                "text_language": "zh",
                "cut_punc": "。",
                "speed": 1.0,
                "top_k": 5,
                "top_p": 1.0,
                "temperature": 0.7
            },
            timeout=60
        )
        if response.status_code == 200:
            with open(tts_file, 'wb') as f:
                f.write(response.content)
            if os.path.getsize(tts_file) > 1024:
                print(f"🔊 正在播放（设备：{PLAY_DEVICE}）...")
                subprocess.run(
                    ["aplay", "-D", PLAY_DEVICE, tts_file],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=30
                )
                return True
            else:
                print("❌ TTS合成失败：返回的音频文件为空")
        else:
            print(f"❌ TTS接口报错：{response.status_code} {response.text}")
    except Exception as e:
        print(f"❌ TTS执行异常：{str(e)}")
    return False

# ===================== 通义千问图片分析（核心修复：从全局缓存取帧）=====================
def analyze_image():
    global latest_frame
    # ✅ 修复：从全局帧缓存里复制帧，不再抢队列里的帧
    with frame_lock:
        if latest_frame is None:
            return "❌ 获取画面失败：暂无有效画面"
        # 复制帧，避免修改原始画面
        frame = latest_frame.copy()
    
    # 保存画面用于AI分析
    cv2.imwrite(CAPTURE_PATH, frame)

    with open(CAPTURE_PATH, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode()

    try:
        headers = {
            "Authorization": f"Bearer {QIANWEN_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": QIANWEN_MODEL,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"text": f"分析这张图片，统计画面中的人数，描述场景内容，人数超过{PEOPLE_LIMIT}人请发出警告"},
                            {"image": f"data:image/jpeg;base64,{img_base64}"}
                        ]
                    }
                ]
            },
            "parameters": {
                "result_format": "message"
            }
        }

        response = requests.post(QIANWEN_API_URL, headers=headers, json=data, timeout=60)
        if response.status_code == 200:
            result = response.json()["output"]["choices"][0]["message"]["content"][0]["text"]
            print(f"🤖 AI分析结果：{result}")
            # 人数超标报警
            people_num = re.findall(r'(\d+)人', result)
            if people_num and int(people_num[0]) > PEOPLE_LIMIT:
                warn_msg = f"警告！当前人数{people_num[0]}人，超过安全阈值{PEOPLE_LIMIT}人！"
                text_to_speech_play(warn_msg)
                return warn_msg + "\n" + result
            text_to_speech_play(result)
            return result
        else:
            return f"❌ API错误：{response.status_code} | {response.text}"
    except Exception as e:
        return f"❌ 分析失败：{str(e)}"

# ===================== 自动分析线程 =====================
def auto_analyze_thread():
    global last_analyze_time
    while not stop_thread:
        now = time.time()
        if now - last_analyze_time >= AUTO_ANALYZE_INTERVAL:
            print("\n🔄 执行定时自动画面分析...")
            analyze_image()
            last_analyze_time = now
        time.sleep(1)

# ===================== YOLO后处理代码 =====================
def dfl(position):
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    y = position.reshape(n, p_num, mc, h, w)
    y = np.exp(y) / np.sum(np.exp(y), axis=2, keepdims=True)
    acc_metrix = np.arange(mc, dtype=np.float32).reshape(1, 1, mc, 1, 1)
    y = np.sum(y * acc_metrix, axis=2)
    return y

def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)
    position = dfl(position)
    box_xy  = grid + 0.5 - position[:,0:2,:,:]
    box_xy2 = grid + 0.5 + position[:,2:4,:,:]
    xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)
    return xyxy

def filter_boxes(boxes, box_confidences, box_class_probs):
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]
    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    return boxes, classes, scores

def nms_boxes(boxes, scores):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    areas = w * h
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[i])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[i])
        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    return np.array(keep)

def post_process(input_data):
    if input_data is None:
        return None, None, None
    boxes, scores, classes_conf = [], [], []
    default_branch = 3
    pair_per_branch = len(input_data) // default_branch
    for i in range(default_branch):
        boxes.append(box_process(input_data[pair_per_branch*i]))
        classes_conf.append(input_data[pair_per_branch*i+1])
        scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0,2,3,1)
        return _in.reshape(-1, ch)

    boxes = np.concatenate([sp_flatten(v) for v in boxes])
    classes_conf = np.concatenate([sp_flatten(v) for v in classes_conf])
    scores = np.concatenate([sp_flatten(v) for v in scores])

    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b, c, s = boxes[inds], classes[inds], scores[inds]
        keep = nms_boxes(b, s)
        if len(keep) > 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])
    if not nboxes:
        return None, None, None
    return np.concatenate(nboxes), np.concatenate(nclasses), np.concatenate(nscores)

def letter_box(im, new_shape, color=(0,0,0)):
    shape = im.shape[:2]
    r = min(new_shape[0]/shape[0], new_shape[1]/shape[1])
    new_unpad = int(round(shape[1]*r)), int(round(shape[0]*r))
    dw, dh = (new_shape[1]-new_unpad[0])/2, (new_shape[0]-new_unpad[1])/2
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left, right = int(round(dw-0.1)), int(round(dw+0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, (top, left), r

def get_real_box(boxes, pad, ratio):
    boxes[:, [0,2]] -= pad[1]
    boxes[:, [1,3]] -= pad[0]
    boxes /= ratio
    return boxes

# ===================== RKNN初始化 =====================
def init_rknn():
    global rknn_lite
    from rknnlite.api import RKNNLite
    rknn_lite = RKNNLite(verbose=False)
    if rknn_lite.load_rknn(RKNN_MODEL_PATH) != 0:
        print("❌ 模型加载失败")
        exit(1)
    if rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
        print("❌ NPU初始化失败")
        exit(1)
    print("✅ RKNN模型加载成功")

# ===================== 不超时的摄像头初始化 =====================
def init_camera_device():
    global camera
    print("正在打开摄像头...")
    for dev_id in CAMERA_DEVICE_CANDIDATES:
        try:
            camera = cv2.VideoCapture(dev_id, cv2.CAP_V4L2)
            if camera.isOpened():
                ret, _ = camera.read()
                if ret:
                    print(f"✅ 摄像头打开成功 /dev/video{dev_id}")
                    return True
                camera.release()
        except:
            pass
    print(f"❌ 摄像头打开失败，已尝试: {CAMERA_DEVICE_CANDIDATES}")
    return False

def camera_reader():
    global stop_thread, latest_frame
    while not stop_thread:
        if camera and camera.isOpened():
            ret, frame = camera.read()
            if ret:
                # ✅ 修复：摄像头线程同步更新全局帧缓存
                with frame_lock:
                    latest_frame = frame.copy()
                # 同时放入队列供主循环使用
                if frame_queue.full():
                    frame_queue.get_nowait()
                frame_queue.put(frame)
        time.sleep(0.005)

# ===================== 点击跟踪API =====================
@app.route('/click', methods=['POST'])
def click_handler():
    global current_state, track_info, current_detections, object_id_counter
    data = request.get_json()
    norm_x = data.get('x', 0.0)
    norm_y = data.get('y', 0.0)

    with state_lock:
        try:
            frame = frame_queue.get_nowait()
            h, w = frame.shape[:2]
            x = int(norm_x * w)
            y = int(norm_y * h)
        except:
            return {"status": "error", "msg": "no frame"}

        clicked_obj_id = None
        for obj_id, (x1, y1, x2, y2, _, _) in current_detections.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                clicked_obj_id = obj_id
                break

        if current_state == "detect":
            if clicked_obj_id is not None:
                x1, y1, x2, y2, cls, score = current_detections[clicked_obj_id]
                bbox = (int(x1), int(y1), int(x2-x1), int(y2-y1))
                tracker = cv2.TrackerKCF_create()
                tracker.init(frame, bbox)

                track_info = {
                    "target_id": clicked_obj_id,
                    "target_bbox": bbox,
                    "tracker": tracker
                }
                current_state = "track"
                return {"status": "ok", "state": "track", "target_id": clicked_obj_id}
            else:
                return {"status": "error", "msg": "no object at click"}

        elif current_state == "track":
            if clicked_obj_id is not None and clicked_obj_id != track_info["target_id"]:
                current_state = "detect"
                track_info = {"target_id": None, "target_bbox": None, "tracker": None}
                return {"status": "ok", "state": "detect", "msg": "switch to detect"}
            else:
                current_state = "detect"
                track_info = {"target_id": None, "target_bbox": None, "tracker": None}
                return {"status": "ok", "state": "detect", "msg": "switch to detect"}

    return {"status": "error"}

# ===================== 绘制函数 =====================
def draw_detection(img, detections, state, target_id=None):
    for obj_id, (x1, y1, x2, y2, cls, score) in detections.items():
        if state == "detect":
            color = (255, 0, 0)
        else:
            if obj_id != target_id:
                continue
            color = (0, 0, 255)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        if cls == -1:
            label = f"ID:{obj_id}"
        else:
            label = f"ID:{obj_id} {CLASSES[int(cls)]} {score:.2f}"
        cv2.putText(img, label, (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(img, f"State: {state.upper()}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

# ===================== 视频流生成 =====================
def generate_stream():
    global current_state, track_info, current_detections, object_id_counter
    last_detect_time = 0
    while True:
        try:
            frame = frame_queue.get(timeout=0.5)
        except:
            continue

        h, w = frame.shape[:2]
        current_time = time.time()

        if current_time - last_detect_time >= DETECT_INTERVAL:
            last_detect_time = current_time
            img_letter, pad, ratio = letter_box(frame.copy(), (IMG_SIZE[1], IMG_SIZE[0]))
            img_rgb = cv2.cvtColor(img_letter, cv2.COLOR_BGR2RGB)
            img_input = img_rgb[np.newaxis, ...]
            try:
                outputs = rknn_lite.inference(inputs=[img_input])
            except:
                outputs = None
            boxes, classes, scores = post_process(outputs)
            if boxes is not None:
                boxes = get_real_box(boxes, pad, ratio)
                with state_lock:
                    current_detections.clear()
                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes[i].astype(int)
                        cls = int(classes[i])
                        score = scores[i]
                        current_detections[object_id_counter] = (x1, y1, x2, y2, cls, score)
                        object_id_counter += 1
            else:
                with state_lock:
                    current_detections.clear()

        with state_lock:
            cur_state = current_state
            cur_track_info = track_info.copy()
        img_draw = frame.copy()

        if cur_state == "detect":
            draw_detection(img_draw, current_detections, "detect")
        else:
            tracker = cur_track_info.get("tracker")
            target_id = cur_track_info.get("target_id")
            if tracker is None:
                with state_lock:
                    current_state = "detect"
                    track_info = {"target_id": None, "target_bbox": None, "tracker": None}
                draw_detection(img_draw, current_detections, "detect")
            else:
                success, bbox = tracker.update(img_draw)
                if success:
                    x, y, bw, bh = [int(v) for v in bbox]
                    x1, y1, x2, y2 = x, y, x+bw, y+bh
                    temp_det = {target_id: (x1, y1, x2, y2, -1, 1.0)}
                    draw_detection(img_draw, temp_det, "track", target_id)
                else:
                    with state_lock:
                        current_state = "detect"
                        track_info = {"target_id": None, "target_bbox": None, "tracker": None}
                    draw_detection(img_draw, current_detections, "detect")

        fps = 1.0 / (time.time() - current_time + 1e-6)
        fps_queue.append(fps)
        cv2.putText(img_draw, f"FPS: {np.mean(fps_queue):.1f}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        ret, jpeg = cv2.imencode('.jpg', img_draw, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ret:
            continue
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n'

# ===================== 手动分析接口 =====================
@app.route('/analyze', methods=['POST'])
def manual_analyze():
    return jsonify(result=analyze_image())

# ===================== 前端页面 =====================
@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>YOLO11 实时检测与跟踪</title>
        <style>
            body{background:#1a1a2e;color:white;text-align:center;padding:20px;font-family:Arial;}
            img{max-width:100%;border-radius:10px;cursor:crosshair;border:2px solid #333;}
            .status{padding:10px;background:#0f3460;display:inline-block;border-radius:8px;margin-bottom:20px;}
            .btn{padding:12px 24px;background:#00ffaa;border:none;border-radius:8px;font-size:16px;margin:10px;cursor:pointer}
            .result{margin-top:20px;padding:15px;background:#0f3460;border-radius:8px;width:80%;margin-left:auto;margin-right:auto;text-align:left;white-space:pre-wrap;}
        </style>
    </head>
    <body>
        <h1>🔍 YOLO11 RKNN 实时检测与跟踪</h1>
        <button class="btn" onclick="manualAnalyze()">🔍 手动分析画面</button>
        <div class="status">✨ 点击目标跟踪 | 按钮AI分析 | 自动语音播报</div>
        <img id="stream" src="/stream" alt="video stream" crossorigin="anonymous">
        <div>状态: <span id="state">detect</span></div>
        <div class="result" id="result"></div>

        <script>
            const img = document.getElementById('stream');
            // 点击跟踪
            img.addEventListener('click', async (e) => {
                const rect = img.getBoundingClientRect();
                const x = (e.clientX - rect.left)/rect.width;
                const y = (e.clientY - rect.top)/rect.height;
                const res = await fetch('/click',{
                    method:'POST',
                    headers:{'Content-Type':'application/json'},
                    body:JSON.stringify({x,y})
                });
                const data = await res.json();
                document.getElementById('state').innerText = data.state;
            });

            // 手动分析
            async function manualAnalyze(){
                document.getElementById('result').innerText = "⏳ 正在分析画面中...";
                const res = await fetch('/analyze',{method:'POST'});
                const data = await res.json();
                document.getElementById('result').innerText = data.result;
            }
        </script>
    </body>
    </html>
    '''

@app.route('/stream')
def video_feed():
    return Response(generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ===================== 启动程序 =====================
if __name__ == '__main__':
    try:
        init_rknn()
        if not init_camera_device():
            exit(1)
        threading.Thread(target=camera_reader, daemon=True).start()
        threading.Thread(target=auto_analyze_thread, daemon=True).start()
        time.sleep(1)
        print(f"\n🌐 正确访问地址：http://192.168.137.101:{WEB_PORT}")
        print("⚠️  请复制完整地址到浏览器访问，必须加http://和端口号")
        app.run(host='0.0.0.0', port=WEB_PORT, debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        stop_thread = True
    finally:
        if camera:
            camera.release()
        if rknn_lite:
            rknn_lite.release()
        cv2.destroyAllWindows()
        print("✅ 安全退出")
