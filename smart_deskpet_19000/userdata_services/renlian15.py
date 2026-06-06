import os
import cv2
import sys
import time
import numpy as np
import threading
import queue
import glob
from flask import Flask, Response, jsonify, request
from collections import deque

# ===================== 核心配置 =====================
RKNN_MODEL_PATH = '/userdata/bestrenlian22.rknn'
FACE_DIR = "face_db_aligned"
WEB_PORT = 8088

DETECTION_INTERVAL = 0.1  # 降低推理间隔，保证数据实时性
OBJ_THRESH = 0.3
KEYPOINT_THRESH = 0.7
NMS_THRESH = 0.45
IMG_SIZE = (224, 224)
RENDER_W, RENDER_H = 640, 480

# ===================== 全局变量 =====================
app = Flask(__name__)
camera = None
frame_queue = queue.Queue(maxsize=2)
stop_thread = False
last_detect_time = 0
current_boxes = None
current_keypoints = None
thread_lock = threading.Lock()

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

recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=16, grid_x=4, grid_y=4, threshold=80)
name_list = []
capture_mode = False
capture_user = ""
capture_count = 0
latest_frame_data = None

# ===================== 光照预处理 =====================
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

# ===================== 人脸后处理 =====================
def post_process(input_data):
    if input_data is None or len(input_data) == 0:
        return None, None
    raw_output = input_data[0]
    output = raw_output.squeeze(0)
    if output.shape[0] == 20:
        output = output.T
    num_preds = output.shape[0]
    if num_preds == 0:
        return None, None

    x_center = output[:, 0]
    y_center = output[:, 1]
    w = output[:, 2]
    h = output[:, 3]
    box_scores = output[:, 4]
    keypoints = output[:, 5:].reshape(-1, 5, 3)

    keep_indices = box_scores >= OBJ_THRESH
    if not np.any(keep_indices):
        return None, None

    valid_indices = []
    for i in range(len(x_center[keep_indices])):
        kp = keypoints[keep_indices][i]
        if np.any(kp[:, 2] < KEYPOINT_THRESH): continue
        if abs(kp[0, 1] - kp[1, 1]) > 15: continue
        if kp[2, 1] < max(kp[0, 1], kp[1, 1]) + 5: continue
        if kp[3, 1] < kp[2, 1] + 5 or kp[4, 1] < kp[2, 1] + 5: continue
        box_w, box_h = w[keep_indices][i], h[keep_indices][i]
        if abs(box_w / box_h - 1.0) > 0.4: continue
        if box_w * box_h < 400: continue
        valid_indices.append(i)

    if not valid_indices:
        return None, None

    x_center = x_center[keep_indices][valid_indices]
    y_center = y_center[keep_indices][valid_indices]
    w = w[keep_indices][valid_indices]
    h = h[keep_indices][valid_indices]
    box_scores = box_scores[keep_indices][valid_indices]
    keypoints = keypoints[keep_indices][valid_indices]

    x1 = x_center - w / 2
    y1 = y_center - h / 2
    x2 = x_center + w / 2
    y2 = y_center + h / 2

    boxes_for_nms = np.column_stack((x1, y1, w, h))
    keep = cv2.dnn.NMSBoxes(boxes_for_nms.tolist(), box_scores.tolist(), OBJ_THRESH, NMS_THRESH)
    if len(keep) > 0:
        k = keep.flatten()
        return np.column_stack((x1[k], y1[k], x2[k], y2[k])), keypoints[k]
    return None, None

# ===================== RKNN模型初始化 =====================
def load_rknn_model():
    from rknnlite.api import RKNNLite
    global rknn_lite
    rknn_lite = RKNNLite(verbose=False)
    rknn_lite.load_rknn(RKNN_MODEL_PATH)
    rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    print("✅ 模型加载完成")

# ===================== 摄像头线程 =====================
def init_camera_device():
    global camera
    for dev_id in CAMERA_DEVICE_CANDIDATES:
        try:
            camera = cv2.VideoCapture(dev_id, cv2.CAP_V4L2)
            if camera.isOpened():
                ret, _ = camera.read()
                if ret:
                    print(f"✅ 摄像头打开成功 /dev/video{dev_id}")
                    return True
                camera.release()
        except: pass
    print(f"❌ 摄像头打开失败，已尝试: {CAMERA_DEVICE_CANDIDATES}")
    return False

def camera_reader():
    global stop_thread
    while not stop_thread:
        if camera and camera.isOpened():
            ret, frame = camera.read()
            if ret:
                if frame_queue.full(): frame_queue.get_nowait()
                frame_queue.put(frame)
        time.sleep(0.005)

# ===================== 人脸库管理 =====================
def align_face(img, keypoints):
    template_points = np.array([[50,50],[100,50],[75,75],[50,100],[100,100]], dtype=np.float32)
    current_points = keypoints[:, :2].astype(np.float32)
    M, _ = cv2.estimateAffinePartial2D(current_points, template_points)
    return cv2.warpAffine(img, M, (150, 150), flags=cv2.INTER_CUBIC)

def load_faces():
    global name_list, recognizer
    faces, labels, name_list = [], [], []
    if not os.path.exists(FACE_DIR):
        os.makedirs(FACE_DIR, exist_ok=True)
        return
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=16, grid_x=4, grid_y=4, threshold=80)
    for img_name in os.listdir(FACE_DIR):
        if not img_name.endswith(".jpg"): continue
        user = img_name.split("_")[0]
        if user not in name_list: name_list.append(user)
        label_id = name_list.index(user)
        img = cv2.imread(os.path.join(FACE_DIR, img_name), cv2.IMREAD_GRAYSCALE)
        faces.append(img)
        labels.append(label_id)
    if faces: recognizer.train(faces, np.array(labels))

def get_user_list():
    users = {}
    if not os.path.exists(FACE_DIR): return users
    for f in os.listdir(FACE_DIR):
        if f.endswith(".jpg"):
            user = f.split("_")[0]
            users[user] = users.get(user, 0) + 1
    return users

# 【修复】拍照函数：加详细日志 + 容错处理
def capture_single_photo():
    global capture_count, capture_user, capture_mode, latest_frame_data
    print(f"📸 拍照请求：capture_mode={capture_mode}, count={capture_count}, latest_data={latest_frame_data is not None}")
    
    if not capture_mode:
        return {"status":"failed","msg":"未进入录入模式，请先点击开始录入"}
    if capture_count >= 20:
        return {"status":"failed","msg":"已拍满20张"}
    if latest_frame_data is None:
        return {"status":"failed","msg":"未检测到人脸，请调整位置"}
    
    gray_frame, keypoints = latest_frame_data
    if keypoints is None or len(keypoints) == 0:
        return {"status":"failed","msg":"未检测到有效人脸关键点"}
    
    try:
        aligned_face = align_face(gray_frame, keypoints[0])
        aligned_face = cv2.equalizeHist(aligned_face)
        capture_count += 1
        filename = f"{FACE_DIR}/{capture_user}_{capture_count}.jpg"
        cv2.imwrite(filename, aligned_face)
        print(f"✅ 已保存：{filename}")
        
        if capture_count >= 20:
            capture_mode = False
            load_faces()
            return {"status":"done","count":20}
        return {"status":"ok","count":capture_count}
    except Exception as e:
        print(f"❌ 保存失败：{str(e)}")
        return {"status":"failed","msg":f"保存失败：{str(e)}"}

# ===================== 视频流生成 =====================
def generate_video_stream():
    global current_boxes, current_keypoints, last_detect_time, latest_frame_data
    threading.Thread(target=camera_reader, daemon=True).start()
    load_faces()
    sw, sh = RENDER_W/IMG_SIZE[0], RENDER_H/IMG_SIZE[1]
    fps_queue = deque(maxlen=30)
    last_time = time.time()

    while True:
        try: frame = frame_queue.get(timeout=0.5)
        except: continue

        now = time.time()
        # 降低推理间隔，保证数据实时性
        if now - last_detect_time > DETECTION_INTERVAL:
            with thread_lock:
                prep = preprocess_image(frame)
                rgb = cv2.cvtColor(prep, cv2.COLOR_BGR2RGB)
                inp = np.expand_dims(cv2.resize(rgb, IMG_SIZE),0)
                out = rknn_lite.inference(inputs=[inp])
                current_boxes, current_keypoints = post_process(out)
            last_detect_time = now

        disp = frame.copy()
        gray = cv2.cvtColor(disp, cv2.COLOR_BGR2GRAY)
        
        # 【关键修复】确保latest_frame_data实时更新
        if current_boxes is not None and len(current_boxes) > 0 and current_keypoints is not None:
            latest_frame_data = (gray, current_keypoints)
        else:
            latest_frame_data = None

        # FPS计算
        fps = 1/(now-last_time)
        last_time=now
        fps_queue.append(fps)
        avg_fps = sum(fps_queue)/len(fps_queue) if fps_queue else 0

        if current_boxes is not None:
            for i in range(len(current_boxes)):
                x1,y1,x2,y2 = current_boxes[i]
                x1r,y1r,x2r,y2r = int(x1*sw),int(y1*sh),int(x2*sw),int(y2*sh)
                cv2.rectangle(disp,(x1r,y1r),(x2r,y2r),(0,255,0),2)
                kps = current_keypoints[i]
                for j in range(5):
                    kx,ky = int(kps[j,0]*sw),int(kps[j,1]*sh)
                    if kps[j,2]>KEYPOINT_THRESH:
                        cv2.circle(disp,(kx,ky),4,(0,255,255),-1)
                if not capture_mode:
                    try:
                        ag = align_face(gray,kps)
                        ag = cv2.equalizeHist(ag)
                        idx,conf = recognizer.predict(ag)
                        sim = max(0,100-conf)
                        name = name_list[idx] if (idx<len(name_list) and conf<80) else "Unknown"
                        col = (0,255,0) if name!="Unknown" else (0,0,255)
                        cv2.putText(disp,f"{name} {sim:.1f}%",(x1r,y1r-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,col,2)
                    except: pass

        # 【修复】中文乱码：用英文显示录入进度
        if capture_mode:
            cv2.putText(disp,f"Capturing: {capture_count}/20",(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        cv2.putText(disp,f"FPS:{avg_fps:.1f}",(20,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        _, jpg = cv2.imencode('.jpg',disp,[cv2.IMWRITE_JPEG_QUALITY,85])
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+jpg.tobytes()+b'\r\n'

# ===================== Flask路由 =====================
@app.route('/')
def index():
    return '''
<html>
<head>
    <meta charset="UTF-8">
    <title>人脸识别系统</title>
    <style>
        body{background:#1a1a2e;color:white;text-align:center;font-family:Arial;padding:20px;}
        .container{max-width:800px;margin:0 auto;}
        .btn{padding:12px 24px;font-size:16px;margin:8px;border:none;border-radius:8px;cursor:pointer;}
        .btn-green{background:#00b966;color:white;}
        .btn-red{background:#e63946;color:white;}
        .btn-blue{background:#0096c7;color:white;}
        .user-list{margin:20px 0;display:flex;flex-wrap:wrap;justify-content:center;gap:12px;}
        .user-card{background:#252941;padding:12px 18px;border-radius:10px;min-width:120px;}
        .user-card p{margin:4px 0;font-size:14px;}
        input{padding:10px 15px;font-size:16px;width:200px;border-radius:8px;border:none;margin:10px 0;}
        #status{color:#ffb703;margin:10px 0;}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 人脸识别系统</h1>
        <input id="username" placeholder="输入用户名">
        <br>
        <button class="btn btn-blue" onclick="startCapture()">开始录入人脸</button>
        <button class="btn btn-green" onclick="takePhoto()">📸 拍照</button>
        <p id="status">等待操作...</p>

        <h3>👥 已录入用户列表（点击删除）</h3>
        <div id="userList" class="user-list"></div>

        <img src="/video_feed" width="640" style="border-radius:10px;margin-top:20px;">
    </div>

<script>
function loadUserList(){
    fetch('/get_user_list').then(res=>res.json()).then(data=>{
        let list = document.getElementById('userList');
        list.innerHTML = '';
        for(let user in data){
            let card = document.createElement('div');
            card.className = 'user-card';
            card.innerHTML = `
                <p><strong>${user}</strong></p>
                <p>${data[user]} 张照片</p>
                <button class="btn btn-red" onclick="delUser('${user}')">删除</button>
            `;
            list.appendChild(card);
        }
    });
}

function startCapture(){
    let u = document.getElementById('username').value.trim();
    if(!u){alert('请输入用户名！');return;}
    fetch('/start_capture?u='+u).then(()=>{
        document.getElementById('status').textContent = '已进入录入模式，点击拍照';
    });
}

function takePhoto(){
    fetch('/capture_photo').then(res=>res.json()).then(d=>{
        if(d.status==='ok'){
            document.getElementById('status').textContent = `已拍 ${d.count}/20 张`;
        }else if(d.status==='done'){
            document.getElementById('status').textContent = '✅ 录入完成！';
            loadUserList();
        }else{
            alert(d.msg);
        }
    });
}

function delUser(user){
    if(!confirm('确定删除用户 '+user+' 吗？')) return;
    fetch('/delete_user?u='+user).then(()=>{
        loadUserList();
        alert('已删除 '+user);
    });
}

window.onload = loadUserList;
</script>
</body>
</html>
'''

@app.route('/get_user_list')
def get_users():
    return jsonify(get_user_list())

@app.route('/start_capture')
def start_cap():
    global capture_mode,capture_user,capture_count
    capture_user = request.args.get('u')
    capture_mode = True
    capture_count = 0
    print(f"✅ 开始录入：用户={capture_user}")
    return jsonify({"status":"ok"})

@app.route('/capture_photo')
def cap_photo():
    return jsonify(capture_single_photo())

@app.route('/delete_user')
def del_user():
    u = request.args.get('u')
    if os.path.exists(FACE_DIR):
        for f in os.listdir(FACE_DIR):
            if f.startswith(f"{u}_"):
                os.remove(os.path.join(FACE_DIR,f))
    load_faces()
    return jsonify({"status":"ok"})

@app.route('/video_feed')
def video():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ===================== 主程序 =====================
if __name__ == '__main__':
    os.makedirs(FACE_DIR, exist_ok=True)
    load_rknn_model()
    init_camera_device()
    print(f"\n🌐 访问: http://192.168.137.101:{WEB_PORT}")
    try:
        app.run(host='0.0.0.0',port=WEB_PORT,debug=False,threaded=True,use_reloader=False)
    except KeyboardInterrupt:
        stop_thread=True
    finally:
        if camera: camera.release()
        if rknn_lite: rknn_lite.release()
        print("✅ 已退出")
