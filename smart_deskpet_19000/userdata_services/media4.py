import cv2
import numpy as np
import mediapipe as mp
import math
import time
import threading
import queue
import os
from flask import Flask, Response
from PIL import Image, ImageDraw, ImageFont

# ===================== 配置 =====================
WEB_PORT = 5000
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

# ===================== 全局 =====================
app = Flask(__name__)
hands = None
mp_drawing = None
mp_hands = None
camera = None
current_gesture = "普通手势"
last_detect_time = 0
thread_lock = threading.Lock()
frame_queue = queue.Queue(maxsize=2)
stop_thread = False
camera_reader_started = False

# ===================== 中文字体修复（核心！） =====================
def cv2_put_text_cn(img, text, pos, font_size=40, color=(0,255,0)):
    try:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", font_size)
        draw.text(pos, text, font=font, fill=color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except:
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        return img

# ===================== 你的手势函数（原样保留） =====================
def cal(p1,p2,w,h):
    x1=int(p1.x*w)
    y1=int(p1.y*h)
    x2=int(p2.x*w)
    y2=int(p2.y*h)
    return math.hypot(x1-x2,y1-y2)

def is_palm_facing(landmark):
    return abs(landmark[0].x - landmark[9].x) > 0.05

def OKfigure(landmark,wid,hig):
    tip1=landmark[4]
    tip2=landmark[8]
    tip3=landmark[12]
    tip4=landmark[16]
    tip5=landmark[20]
    distance=cal(tip1,tip2,wid,hig)
    dis_true=distance<50 and distance>0
    if(tip3.y<landmark[9].y and tip4.y < landmark[13].y and tip5.y < landmark[17].y):
        return dis_true
    return False

wave_history = []
WAVE_BUFFER_SIZE = 10
def wave_figure(landmark,wid,hig):
    global wave_history
    fingers_straight = (landmark[8].y<landmark[6].y and landmark[12].y<landmark[10].y
                       and landmark[16].y<landmark[14].y and landmark[20].y<landmark[18].y)
    if not fingers_straight:
        wave_history.clear()
        return False
    x=int(landmark[0].x * wid)
    wave_history.append(x)
    if len(wave_history)>WAVE_BUFFER_SIZE:
        wave_history.pop(0)
    if len(wave_history)==10:
        return max(wave_history)-min(wave_history)>30
    return False

def is_fist(landmark):
    index_fist = landmark[8].y > landmark[6].y
    middle_fist = landmark[12].y > landmark[10].y
    ring_fist = landmark[16].y > landmark[14].y
    pinky_fist = landmark[20].y > landmark[18].y
    four_fist = index_fist and middle_fist and ring_fist and pinky_fist
    if not four_fist: return False
    thumb_bent = landmark[4].y > landmark[3].y
    thumb_side = abs(landmark[4].x-landmark[6].x)>0.08 and landmark[4].y>landmark[8].y-0.05
    return thumb_bent or thumb_side

def yes_figure(landmark,wid,hig):
    tip1,tip2,tip3,tip4,tip5=landmark[4],landmark[8],landmark[12],landmark[16],landmark[20]
    tip6,tip7=landmark[7],landmark[13]
    x1,x2,x3=int(tip1.x*wid),int(tip4.x*wid),int(tip5.x*wid)
    y1,y2,y3=int(tip1.y*hig),int(tip4.y*hig),int(tip5.y*hig)
    ok = abs(x1-x2)<40 and abs(x1-x3)<40 and abs(x2-x3)<40 and abs(y1-y2)<40
    return ok and tip2.y<tip6.y and tip3.y<tip7.y

fight_history = []
def fight(landmark,wid,hig):
    global fight_history
    if not is_fist(landmark): return False
    x=int(landmark[0].x*wid)
    fight_history.append(x)
    if len(fight_history)>8: fight_history.pop(0)
    return len(fight_history)==8 and (max(fight_history)-min(fight_history))>50

def zan_figure(landmark, wid, hig):
    four = (landmark[8].y>landmark[6].y and landmark[12].y>landmark[10].y
            and landmark[16].y>landmark[14].y and landmark[20].y>landmark[18].y)
    thumb_up = landmark[4].y<landmark[3].y and landmark[3].y<landmark[2].y
    highest = landmark[4].y<landmark[12].y-0.1
    return four and thumb_up and highest

def love_figure(lm1, lm2, w, h):
    d1=cal(lm1[4],lm2[4],w,h)
    d2=cal(lm1[8],lm2[8],w,h)
    d3=cal(lm1[4],lm2[8],w,h)
    d4=cal(lm1[8],lm2[4],w,h)
    return (d1<120 and d2<120) or (d3<120 and d4<120)

# ===================== 初始化（已修复！） =====================
def init_mediapipe():
    global hands, mp_drawing, mp_hands
    mp_hands = mp.solutions.hands
    # ✅ 修复参数格式
    hands = mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=2
    )
    mp_drawing = mp.solutions.drawing_utils
    print("✅ MediaPipe 初始化完成")

def init_camera_device():
    global camera
    for device in CAMERA_DEVICE_CANDIDATES:
        try:
            camera = cv2.VideoCapture(device, cv2.CAP_V4L2)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH,640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
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

# ===================== 摄像头线程 =====================
def camera_reader():
    global stop_thread
    while not stop_thread:
        if camera and camera.isOpened():
            ret, f = camera.read()
            if ret:
                if frame_queue.full(): frame_queue.get_nowait()
                frame_queue.put(f)
        time.sleep(0.005)


def placeholder_frame(text="Waiting for camera frame..."):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (26, 26, 46)
    cv2.putText(frame, text, (36, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (120, 255, 160), 2)
    ok, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jpg.tobytes() if ok else b''

# ===================== 手势识别 =====================
def run_gesture(frame):
    h,w,_ = frame.shape
    txt = "普通手势"
    res = hands.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    if res.multi_hand_landmarks:
        n = len(res.multi_hand_landmarks)
        for hl in res.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame,hl,mp_hands.HAND_CONNECTIONS)
        if n==1:
            lm = res.multi_hand_landmarks[0].landmark
            if OKfigure(lm,w,h): txt="OK手势 ✔️"
            elif fight(lm,w,h): txt="挥拳动作 ✊"
            elif is_fist(lm): txt="握拳动作 👊"
            elif zan_figure(lm,w,h): txt="大拇指动作 👍"
            elif yes_figure(lm,w,h): txt="yes手势 👌"
            elif wave_figure(lm,w,h): txt="挥手动作 👋"
        if n==2:
            if love_figure(res.multi_hand_landmarks[0].landmark,
                           res.multi_hand_landmarks[1].landmark,w,h):
                txt="比心 ❤️"
            else: txt="双手检测中..."
    return frame, txt

# ===================== 推流 =====================
def generate_video_stream():
    global current_gesture, last_detect_time, camera_reader_started
    if not (camera and camera.isOpened()):
        init_camera_device()
    if not camera_reader_started:
        threading.Thread(target=camera_reader,daemon=True).start()
        camera_reader_started = True
    idle_since = time.time()
    while True:
        try:
            frame = frame_queue.get(timeout=0.5)
            idle_since = time.time()
        except:
            if time.time() - idle_since > 2.0:
                init_camera_device()
                idle_since = time.time()
            jpg = placeholder_frame()
            if jpg:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n'
            continue
        t = time.time()
        if t - last_detect_time > DETECTION_INTERVAL:
            with thread_lock:
                frame, current_gesture = run_gesture(frame)
            last_detect_time = t
        # 绘制中文
        frame = cv2_put_text_cn(frame, current_gesture, (20,50), 40, (0,255,0))
        _, jpg = cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,85])
        yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+jpg.tobytes()+b'\r\n'

# ===================== 网页 =====================
@app.route('/')
def index():
    return '''
<html>
<head>
<meta charset="UTF-8">
<title>手势识别</title>
<style>body{background:#1a1a2e;text-align:center;color:white}
img{border:5px solid #00ff88;border-radius:10px}</style>
</head>
<body><h1>🤚 手势识别系统</h1><img src="/video_feed?t=1" width="640"></body></html>
'''

@app.route('/video_feed')
def feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ===================== 主程序 =====================
if __name__ == '__main__':
    print("="*60)
    print("  手势识别系统（中文正常显示）")
    print("="*60)
    init_mediapipe()
    init_camera_device()
    print(f"🌐 访问：http://192.168.137.101:{WEB_PORT}")
    try:
        app.run(host='0.0.0.0',port=WEB_PORT,debug=False,threaded=True,use_reloader=False)
    except KeyboardInterrupt:
        stop_thread=True
    finally:
        camera.release()
        hands.close()
        print("✅ 已退出")
