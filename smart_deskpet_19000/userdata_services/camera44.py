import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
from rknnlite.api import RKNNLite
import time
import threading
import queue
import requests
import os
import math

# ===================== 核心配置 =====================
RKNN_MODEL = '/home/linaro/bestxuboran.rknn'
WEB_PORT = 8088
DETECTION_INTERVAL = 1.0          # 全局检测间隔（秒）
INPUT_SIZE = 640
NUM_CLASSES = 1
CONFIDENCE_THRESHOLD = 0.7
NMS_THRESH = 0.15

# TTS 配置（可选）
TTS_API_URL = "http://127.0.0.1:9880/"
TTS_OUTPUT_DIR = "/userdata/tts_out"
PLAY_COMMAND = "aplay -D plughw:3,0 /userdata/tts_out/full_result.wav"

# 跟踪与轨迹配置
TRACKER_TYPE = 'CSRT'             # 'KCF' 或 'CSRT'
MAX_HISTORY = 50                  # 最大保存轨迹点数（包括预测）
PREDICT_STEPS = 10                # 预测未来帧数（用于显示）
PREDICT_ALPHA = 0.7               # 卡尔曼过程噪声系数
LOST_TIMEOUT = 3.0                # 丢失超时秒数，超时后彻底放弃切回 detect
RE_DETECT_INTERVAL = 0.5          # 丢失状态下重检测间隔
RE_DETECT_IOU_THRESH = 0.3        # 重检测匹配的 IoU 阈值
RE_DETECT_DIST_THRESH = 100       # 重检测位置距离阈值（像素）
# ===================================================

app = Flask(__name__)
rknn_lite = None
camera = None
frame_queue = queue.Queue(maxsize=2)
stop_thread = False
os.makedirs(TTS_OUTPUT_DIR, exist_ok=True)
is_busy = False

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

# 全局变量
object_id_counter = 1
current_detections = {}            # {id: (x1,y1,x2,y2,score)}
current_state = "detect"           # "detect" or "track"
# track_info 扩展
track_info = {
    "target_id": None,
    "tracker": None,
    "track_state": "inactive",      # inactive, active, lost
    "last_bbox": None,              # (x,y,w,h)
    "history_centers": [],          # 连续轨迹点（包括真实和预测）
    "kalman": None,
    "lost_start_time": 0,
    "last_re_detect_time": 0,
    "predicted_center": None
}
state_lock = threading.Lock()
latest_frame = None

# ===================== TTS 函数（不变） =====================
def text_to_speech(text):
    global is_busy
    if is_busy:
        return False
    is_busy = True
    try:
        print(f"🗣️ 播报：{text}")
        data = {"text": text, "text_language": "zh", "cut_punc": "。"}
        response = requests.post(TTS_API_URL, json=data, timeout=60)
        if response.status_code == 200:
            with open(f"{TTS_OUTPUT_DIR}/full_result.wav", 'wb') as f:
                f.write(response.content)
            os.system(PLAY_COMMAND)
            return True
    except Exception as e:
        print(f"❌ 播报失败：{str(e)}")
    finally:
        is_busy = False
    return False

# ===================== 图像预处理 =====================
def analyze_light_condition(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_bright = np.mean(gray)
    dark_ratio = np.sum(gray < 50) / gray.size * 100
    if mean_bright > 150 and dark_ratio > 30:
        return "逆光"
    elif mean_bright < 60:
        return "昏暗"
    elif mean_bright > 220:
        return "过曝"
    else:
        return "正常"

def preprocess_image(frame):
    denoised = cv2.GaussianBlur(frame, (3, 3), 0)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_img = cv2.merge((l_enhanced, a, b))
    enhanced_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
    avg_bright = np.mean(gray)
    gamma = 1.0
    if avg_bright < 60:
        gamma = 1.7
    elif avg_bright > 220:
        gamma = 0.5
    inv_gamma = 1.0 / gamma
    gamma_table = np.array([((i / 255.0) ** inv_gamma)*255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(enhanced_img, gamma_table)

def nms(boxes, scores, threshold=NMS_THRESH):
    if len(boxes) == 0:
        return [], []
    boxes = np.array(boxes)
    scores = np.array(scores)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]
    return boxes[keep].tolist(), scores[keep].tolist()

# ===================== RKNN 初始化 =====================
def load_rknn_model():
    global rknn_lite
    print("="*60)
    rknn_lite = RKNNLite(verbose=False)
    ret = rknn_lite.load_rknn(RKNN_MODEL)
    if ret != 0:
        print("❌ 模型加载失败")
        return False
    ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    if ret != 0:
        print("❌ NPU 初始化失败")
        return False
    print("✅ 模型加载完成")
    return True

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
    global stop_thread
    while not stop_thread:
        if camera and camera.isOpened():
            ret, frame = camera.read()
            if ret:
                if frame_queue.full():
                    frame_queue.get_nowait()
                frame_queue.put(frame)
        time.sleep(0.005)

# ===================== 人头检测（返回带ID的字典） =====================
def run_detection(frame, roi=None):
    """如果 roi 不为 None，则只在 roi 区域检测（用于丢失恢复时的局部检测）"""
    if rknn_lite is None:
        return {}
    try:
        frame_proc = preprocess_image(frame)
        orig_h, orig_w = frame_proc.shape[:2]

        # 如果指定 ROI，则裁剪区域
        if roi is not None:
            rx1, ry1, rx2, ry2 = roi
            rx1 = max(0, rx1)
            ry1 = max(0, ry1)
            rx2 = min(orig_w, rx2)
            ry2 = min(orig_h, ry2)
            if rx2 <= rx1 or ry2 <= ry1:
                return {}
            frame_proc = frame_proc[ry1:ry2, rx1:rx2]
            offset_x, offset_y = rx1, ry1
        else:
            offset_x, offset_y = 0, 0

        img_rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
        h_new, w_new = frame_proc.shape[:2]

        scale = min(INPUT_SIZE / w_new, INPUT_SIZE / h_new)
        new_w, new_h = int(w_new * scale), int(h_new * scale)
        resized = cv2.resize(img_rgb, (new_w, new_h))

        pad_w = INPUT_SIZE - new_w
        pad_h = INPUT_SIZE - new_h
        padded = cv2.copyMakeBorder(resized, pad_h//2, pad_h - pad_h//2,
                                    pad_w//2, pad_w - pad_w//2,
                                    cv2.BORDER_CONSTANT, value=(114,114,114))
        input_data = np.expand_dims(padded, axis=0)

        outputs = rknn_lite.inference(inputs=[input_data])
        output = outputs[0].reshape(4 + NUM_CLASSES, 8400).T

        boxes, scores = [], []
        light = analyze_light_condition(frame)
        threshold = 0.4 if light != "正常" else CONFIDENCE_THRESHOLD

        for i in range(8400):
            score = np.max(output[i, 4:])
            if score > threshold:
                cx, cy, bw, bh = output[i, :4]
                cx = (cx - pad_w//2) / scale
                cy = (cy - pad_h//2) / scale
                bw /= scale
                bh /= scale
                x1 = max(0, int(cx - bw/2))
                y1 = max(0, int(cy - bh/2))
                x2 = min(w_new, int(cx + bw/2))
                y2 = min(h_new, int(cy + bh/2))
                # 加上偏移量
                x1 += offset_x
                y1 += offset_y
                x2 += offset_x
                y2 += offset_y
                boxes.append([x1, y1, x2, y2])
                scores.append(float(score))

        boxes, scores = nms(boxes, scores)
        detections = {}
        with state_lock:
            global object_id_counter
            for box, score in zip(boxes, scores):
                detections[object_id_counter] = (*box, score)
                object_id_counter += 1
        return detections
    except Exception as e:
        print(f"检测出错: {e}")
        return {}

# ===================== 卡尔曼滤波 =====================
def init_kalman():
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
    kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * PREDICT_ALPHA
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
    return kalman

def kalman_predict(kalman):
    """返回预测的 (cx, cy)"""
    if kalman is None:
        return None
    pred = kalman.predict()
    return int(pred[0,0]), int(pred[1,0])

def kalman_correct(kalman, cx, cy):
    if kalman is None:
        return
    measurement = np.array([[cx], [cy]], dtype=np.float32)
    kalman.correct(measurement)

# ===================== 轨迹绘制与状态显示 =====================
def draw_overlay(img, state, track_info, detections, tracker_bbox=None):
    # 1. 检测状态：画所有检测框
    if state == "detect":
        for obj_id, (x1, y1, x2, y2, score) in detections.items():
            # 可选：过滤有效人头框（宽高比、面积等）
            w_box = x2 - x1
            h_box = y2 - y1
            if w_box <= 0 or h_box <= 0:
                continue
            aspect = w_box / h_box
            if aspect < 0.6 or aspect > 1.5:
                continue   # 不合理的框不画
            area = w_box * h_box
            if area < 900 or area > 40000:
                continue
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"ID:{obj_id} {score:.2f}"
            cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

    else:  # track 状态
        # 绘制历史轨迹（包括真实和预测）
        history = track_info.get("history_centers", [])
        if len(history) > 0:
            # 绘制轨迹点
            for i, pt in enumerate(history):
                # 真实点绿色，预测点黄色（通过标记区分）
                is_real = (track_info.get("track_state") == "active") or (i < len(history)-PREDICT_STEPS)
                color = (0,255,0) if is_real else (0,255,255)
                cv2.circle(img, pt, 3, color, -1)
            # 连线
            for i in range(1, len(history)):
                cv2.line(img, history[i-1], history[i], (0,255,0), 2 if i < len(history)-PREDICT_STEPS else 1)

        # 绘制当前跟踪框（如果 active 且有实际框）
        if track_info.get("track_state") == "active" and tracker_bbox is not None:
            x, y, w, h = tracker_bbox
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
            cv2.putText(img, f"Tracking ID:{track_info['target_id']}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        elif track_info.get("track_state") == "lost":
            # 丢失状态：显示预测框（虚线）
            pred_center = track_info.get("predicted_center")
            if pred_center and track_info["last_bbox"]:
                # 根据预测中心绘制一个估计框（大小沿用最后成功框）
                last_bbox = track_info["last_bbox"]
                w_bbox, h_bbox = last_bbox[2], last_bbox[3]
                x = pred_center[0] - w_bbox//2
                y = pred_center[1] - h_bbox//2
                # 绘制虚线矩形（OpenCV 无直接虚线，可用 rectangle 配合线型）
                cv2.rectangle(img, (x, y), (x+w_bbox, y+h_bbox), (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(img, "LOST (predicted)", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        # 显示跟踪状态文字
        state_text = f"Track State: {track_info.get('track_state', 'unknown').upper()}"
        cv2.putText(img, state_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    # 统一显示人数
    cv2.putText(img, f"People: {len(detections)}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

# ===================== 视频流生成（长时跟踪逻辑） =====================
def generate_video_stream():
    global current_detections, current_state, track_info, latest_frame
    global object_id_counter
    reader_thread = threading.Thread(target=camera_reader, daemon=True)
    reader_thread.start()
    last_detect_time = 0
    last_re_detect_time = 0

    while True:
        try:
            frame = frame_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        latest_frame = frame.copy()
        current_time = time.time()

        # ========== 全局周期性检测（用于更新 current_detections 和人数显示） ==========
        if current_time - last_detect_time > DETECTION_INTERVAL:
            dets = run_detection(frame)
            with state_lock:
                current_detections = dets
            last_detect_time = current_time

        # ========== 跟踪处理 ==========
        with state_lock:
            state = current_state
            t_info = track_info.copy()

        display = frame.copy()
        tracker_bbox = None

        if state == "track":
            tracker = t_info.get("tracker")
            track_state = t_info.get("track_state", "inactive")

            if track_state == "active" and tracker is not None:
                # 正常跟踪
                success, bbox = tracker.update(display)
                if success:
                    tracker_bbox = bbox
                    cx = int(bbox[0] + bbox[2]/2)
                    cy = int(bbox[1] + bbox[3]/2)
                    # 更新卡尔曼
                    kalman = t_info["kalman"]
                    if kalman is None:
                        kalman = init_kalman()
                        kalman.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
                    else:
                        kalman_correct(kalman, cx, cy)
                    # 更新轨迹历史
                    history = t_info["history_centers"] + [(cx, cy)]
                    if len(history) > MAX_HISTORY:
                        history = history[-MAX_HISTORY:]
                    with state_lock:
                        track_info["history_centers"] = history
                        track_info["last_bbox"] = bbox
                        track_info["kalman"] = kalman
                        track_info["track_state"] = "active"
                        track_info["lost_start_time"] = 0
                else:
                    # 跟踪失败，进入丢失状态
                    print("⚠️ 跟踪丢失，进入预测状态")
                    with state_lock:
                        track_info["track_state"] = "lost"
                        track_info["lost_start_time"] = current_time
                        # 保存最后成功框
                        track_info["last_bbox"] = t_info.get("last_bbox", None)
                        # 初始化历史（如果有中心点）
                    # 进入丢失后，下面统一处理预测

            # 处理丢失状态（包括刚进入丢失以及持续丢失）
            with state_lock:
                track_state = track_info["track_state"]
            if track_state == "lost":
                kalman = t_info["kalman"]
                if kalman is not None:
                    # 预测下一中心点
                    pred_center = kalman_predict(kalman)
                    if pred_center:
                        with state_lock:
                            track_info["predicted_center"] = pred_center
                            # 将预测点加入历史轨迹（用于完整显示）
                            history = track_info["history_centers"] + [pred_center]
                            if len(history) > MAX_HISTORY:
                                history = history[-MAX_HISTORY:]
                            track_info["history_centers"] = history
                            # 更新 last_bbox 用于绘制（保持原大小）
                            last_bbox = track_info["last_bbox"]
                            if last_bbox:
                                # 预测框不会用于跟踪器更新，仅用于显示
                                pass
                # 尝试重检测（每隔 RE_DETECT_INTERVAL 秒）
                if current_time - last_re_detect_time > RE_DETECT_INTERVAL:
                    last_re_detect_time = current_time
                    # 确定 ROI: 基于预测中心扩大区域
                    if t_info["predicted_center"] and t_info["last_bbox"]:
                        pcx, pcy = t_info["predicted_center"]
                        wb, hb = t_info["last_bbox"][2], t_info["last_bbox"][3]
                        roi_margin = int(max(wb, hb) * 2)
                        roi = (pcx - roi_margin, pcy - roi_margin, pcx + roi_margin, pcy + roi_margin)
                        # 在 ROI 内运行检测
                        local_dets = run_detection(frame, roi=roi)
                        # 寻找与预测位置最近且 IoU 符合的检测框
                        best_det = None
                        best_iou = 0
                        for det_id, (x1, y1, x2, y2, score) in local_dets.items():
                            det_cx = (x1 + x2) // 2
                            det_cy = (y1 + y2) // 2
                            # 计算距离和 IoU
                            dist = math.hypot(det_cx - pcx, det_cy - pcy)
                            if dist > RE_DETECT_DIST_THRESH:
                                continue
                            # 简单 IoU 计算
                            ix1 = max(x1, pcx - wb//2)
                            iy1 = max(y1, pcy - hb//2)
                            ix2 = min(x2, pcx + wb//2)
                            iy2 = min(y2, pcy + hb//2)
                            if ix2 > ix1 and iy2 > iy1:
                                inter = (ix2-ix1)*(iy2-iy1)
                                box_area = (x2-x1)*(y2-y1)
                                pred_area = wb*hb
                                iou = inter / (box_area + pred_area - inter)
                                if iou > RE_DETECT_IOU_THRESH and iou > best_iou:
                                    best_iou = iou
                                    best_det = (x1, y1, x2, y2, score)
                        if best_det is not None:
                            # 重新初始化跟踪器
                            x1, y1, x2, y2, score = best_det
                            bbox = (x1, y1, x2-x1, y2-y1)
                            if TRACKER_TYPE == 'KCF':
                                new_tracker = cv2.TrackerKCF_create()
                            else:
                                new_tracker = cv2.TrackerCSRT_create()
                            new_tracker.init(frame, bbox)
                            # 重置卡尔曼状态（用新检测框中心）
                            kalman = init_kalman()
                            cx = (x1 + x2)//2
                            cy = (y1 + y2)//2
                            kalman.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
                            # 更新历史轨迹（保留之前轨迹，添加新点）
                            with state_lock:
                                track_info["tracker"] = new_tracker
                                track_info["kalman"] = kalman
                                track_info["track_state"] = "active"
                                track_info["last_bbox"] = bbox
                                track_info["lost_start_time"] = 0
                                # 将新中心点加入历史（可选，保持连续）
                                track_info["history_centers"].append((cx, cy))
                            print(f"✅ 重新捕获目标，ID={track_info['target_id']}")
                            # 本轮已经恢复，跳过后续丢失绘制
                            continue
                # 检查是否超时（丢失太久则放弃跟踪，切回 detect）
                if t_info["lost_start_time"] > 0 and (current_time - t_info["lost_start_time"]) > LOST_TIMEOUT:
                    print("⏱️ 丢失超时，放弃跟踪，回退检测模式")
                    with state_lock:
                        current_state = "detect"
                        track_info = {"target_id": None, "tracker": None, "track_state": "inactive",
                                      "last_bbox": None, "history_centers": [], "kalman": None,
                                      "lost_start_time": 0, "last_re_detect_time": 0, "predicted_center": None}
                    # 继续循环
                    continue

        # ========== 绘制最终画面 ==========
        with state_lock:
            draw_overlay(display, current_state, track_info, current_detections, tracker_bbox)

        # 编码输出
        _, jpeg = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

# ===================== 后端点击接口（添加有效性过滤） =====================
@app.route('/click', methods=['POST'])
def click_handler():
    global latest_frame, current_state, track_info, current_detections
    data = request.get_json()
    norm_x = data.get('x', 0.0)
    norm_y = data.get('y', 0.0)

    if latest_frame is None:
        return jsonify({"success": False, "msg": "未就绪"})

    h, w = latest_frame.shape[:2]
    px = int(norm_x * w)
    py = int(norm_y * h)
    print(f"📌 点击坐标: ({px},{py})")

    with state_lock:
        if not current_detections:
            return jsonify({"success": False, "msg": "当前无检测结果", "state": "detect"})

        # 过滤有效人头
        def is_valid_head(box, score):
            x1, y1, x2, y2 = box
            wb = x2 - x1
            hb = y2 - y1
            if wb <= 0 or hb <= 0:
                return False
            aspect = wb / hb
            if aspect < 0.6 or aspect > 1.5:
                return False
            area = wb * hb
            if area < 900 or area > 40000:
                return False
            if score < 0.5:
                return False
            return True

        valid = {}
        for obj_id, (x1, y1, x2, y2, score) in current_detections.items():
            if is_valid_head((x1, y1, x2, y2), score):
                valid[obj_id] = (x1, y1, x2, y2, score)

        if not valid:
            return jsonify({"success": False, "msg": "没有有效的人头", "state": "detect"})

        # 找最近的
        best_id = None
        best_dist = float('inf')
        for obj_id, (x1, y1, x2, y2, _) in valid.items():
            cx = (x1 + x2)//2
            cy = (y1 + y2)//2
            dist = (cx - px)**2 + (cy - py)**2
            if dist < best_dist:
                best_dist = dist
                best_id = obj_id

        if best_id is None or best_dist > 3000:
            return jsonify({"success": False, "msg": "未点中有效人头", "state": "detect"})

        x1, y1, x2, y2, score = valid[best_id]
        bbox = (x1, y1, x2-x1, y2-y1)
        if TRACKER_TYPE == 'KCF':
            tracker = cv2.TrackerKCF_create()
        else:
            tracker = cv2.TrackerCSRT_create()
        tracker.init(latest_frame, bbox)

        kalman = init_kalman()
        cx = (x1 + x2)//2
        cy = (y1 + y2)//2
        kalman.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)

        track_info = {
            "target_id": best_id,
            "tracker": tracker,
            "track_state": "active",
            "last_bbox": bbox,
            "history_centers": [(cx, cy)],
            "kalman": kalman,
            "lost_start_time": 0,
            "last_re_detect_time": 0,
            "predicted_center": (cx, cy)
        }
        current_state = "track"
        print(f"✅ 开始跟踪 ID={best_id}")
        return jsonify({"success": True, "state": "track", "target_id": best_id})

# ===================== 网页前端（略作修改，显示状态） =====================
@app.route('/')
def index_page():
    return '''
<html>
<head>
    <meta charset="UTF-8">
    <title>长时人头跟踪 + 轨迹预测</title>
    <style>
        body{background:#1a1a2e;text-align:center;color:white;font-family:Arial;padding:20px;}
        .btn{background:#00ff88;color:#000;padding:15px 40px;font-size:20px;border:none;border-radius:10px;cursor:pointer;margin:20px;}
        img{border:5px solid #00ff88;border-radius:10px;cursor:crosshair;max-width:90%;}
        .info{margin-top:15px;background:#0f3460;display:inline-block;padding:10px;border-radius:10px;}
        .status{font-size:18px;margin-top:10px;}
    </style>
</head>
<body>
    <h1>🧠 人头长时跟踪（丢失后仍预测轨迹）</h1>
    <button class="btn" id="announceBtn" onclick="announce()">🔊 播报人数</button>
    <br>
    <img id="videoImg" src="/video_feed" width="800">
    <div class="info">💡 点击人头开始跟踪。丢失后系统会继续预测轨迹，并在重新出现时自动接续。</div>
    <div class="status" id="stateMsg">状态: 检测模式</div>
    <script>
        const img = document.getElementById('videoImg');
        let lastClick = 0;
        img.addEventListener('click', async (e) => {
            if (Date.now() - lastClick < 300) return;
            lastClick = Date.now();
            const rect = img.getBoundingClientRect();
            const x = (e.clientX - rect.left) / rect.width;
            const y = (e.clientY - rect.top) / rect.height;
            const statusDiv = document.getElementById('stateMsg');
            statusDiv.innerText = "⏳ 正在选择...";
            try {
                const resp = await fetch('/click', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({x: x, y: y})
                });
                const data = await resp.json();
                if (data.success) {
                    statusDiv.innerText = `状态: 跟踪模式 (目标ID ${data.target_id})`;
                } else {
                    statusDiv.innerText = `❌ ${data.msg}`;
                    setTimeout(() => statusDiv.innerText = "状态: 检测模式", 2000);
                }
            } catch(err) {
                statusDiv.innerText = "网络错误";
            }
        });
        async function announce() {
            const btn = document.getElementById('announceBtn');
            if (btn.disabled) return;
            btn.disabled = true;
            btn.innerText = "⏳ ...";
            await fetch('/tts_announce');
            setTimeout(() => { btn.disabled = false; btn.innerText = "🔊 播报人数"; }, 3000);
        }
    </script>
</body>
</html>
'''

@app.route('/tts_announce')
def tts_announce():
    count = len(current_detections)
    if count == 0:
        text = "当前没有检测到人员"
    elif count == 1:
        text = "当前检测到一个人"
    else:
        text = f"当前检测到{count}个人"
    threading.Thread(target=text_to_speech, args=(text,), daemon=True).start()
    return jsonify({"success": True, "count": count, "text": text})

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("="*60)
    print("  长时人头跟踪 + 轨迹预测系统")
    print("="*60)
    if not load_rknn_model():
        exit(1)
    if not init_camera_device():
        exit(1)
    print(f"\n🌐 访问 http://192.168.137.101:{WEB_PORT}")
    try:
        app.run(host='0.0.0.0', port=WEB_PORT, debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        stop_thread = True
    finally:
        if camera: camera.release()
        if rknn_lite: rknn_lite.release()
        cv2.destroyAllWindows()
        print("✅ 退出")
