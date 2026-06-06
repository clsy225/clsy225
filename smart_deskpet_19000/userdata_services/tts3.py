import requests
import os
import subprocess
import time
import importlib.util
import re

# ===================== 【自动识别音频设备·解决序号乱变】=====================
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
BASE_URL = "https://api.deepseek.com/v1"
TTS_API_URL = "http://127.0.0.1:9880/"
TEMP_DIR = "/home/linaro/voice_temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# 自动识别 杰理UACDemoV1.0 声卡（核心修复）
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

# 自动配置播放/录音设备
AUDIO_CARD = get_usb_audio_card()
if AUDIO_CARD:
    PLAY_DEVICE = f"plughw:{AUDIO_CARD},0"
    RECORD_DEVICE = f"plughw:{AUDIO_CARD},0"
    print(f"✅ 自动匹配杰理USB音频：card{AUDIO_CARD}")
else:
    PLAY_DEVICE = "default"
    RECORD_DEVICE = "default"
    print("⚠️  未找到杰理设备，使用系统默认声卡")

# ===================== 无效内容过滤规则 =====================
USELESS_WORDS = {"这", "那", "的", "了", "啊", "嗯", "哦", "呃", "吧", "吗", "呀", "个", "是"}
MIN_VALID_LENGTH = 2

# ===================== 自动检查安装依赖 =====================
def check_and_install(package, install_name=None):
    if install_name is None:
        install_name = package
    if importlib.util.find_spec(package) is None:
        print(f"📦 正在安装 {package}...")
        subprocess.check_call([
            "python3", "-m", "pip", "install",
            install_name, "-i", "https://mirrors.aliyun.com/pypi/simple/"
        ])

print("🔍 检查语音依赖...")
check_and_install("funasr")
check_and_install("modelscope")
check_and_install("torchaudio")

from funasr import AutoModel

# ===================== 初始化ASR语音识别 =====================
print("\n🔄 加载语音识别模型...")
asr_model = None
try:
    asr_model = AutoModel(model="paraformer-zh", model_revision="v2.0.4")
    print("✅ 语音识别模型加载完成")
except Exception as e:
    print(f"❌ 模型加载失败：{e}")
    print("💡 已进入纯文字对话模式")

# ===================== AI对话接口 =====================
def ask_deepseek(prompt):
    print("🤔 思考中...")
    try:
        response = requests.post(
            f"{BASE_URL}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
            },
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": "你是友好助手，回答简洁口语化，不超过100字"},
                    {"role": "user", "content": prompt}
                ],
                "stream": False
            }
        )
        if response.status_code == 200:
            ai_reply = response.json()["choices"][0]["message"]["content"]
            print(f"🤖 AI：{ai_reply}")
            return ai_reply
        else:
            print(f"❌ API错误：{response.status_code}")
            return "API调用失败"
    except Exception as e:
        print(f"❌ 网络错误：{str(e)}")
        return "网络异常"

# ===================== TTS语音合成播放 =====================
def text_to_speech_play(text):
    tts_file = os.path.join(TEMP_DIR, "tts_reply.wav")
    if os.path.exists(tts_file):
        os.remove(tts_file)
    try:
        print("🔊 合成语音中...")
        response = requests.post(TTS_API_URL, json={
            "text": text,
            "text_language": "zh",
            "cut_punc": "。"
        })
        if response.status_code == 200:
            with open(tts_file, 'wb') as f:
                f.write(response.content)
            print("🔊 正在播放...")
            subprocess.run(
                ["aplay", "-D", PLAY_DEVICE, tts_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            print(f"❌ TTS合成失败：{response.status_code}")
    except Exception as e:
        print(f"❌ TTS错误：{str(e)}")

# ===================== 录音功能 =====================
def record_audio():
    record_file = os.path.join(TEMP_DIR, "record.wav")
    if os.path.exists(record_file):
        os.remove(record_file)
    input("\n🎤 按回车开始说话，说完再按回车结束...")
    print("🎙️  正在录音...")
    proc = subprocess.Popen(
        ["arecord", "-D", RECORD_DEVICE, "-f", "S16_LE", "-r", "16000", "-c", "1", record_file],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    input("按回车结束录音...")
    proc.terminate()
    proc.wait()
    time.sleep(0.5)
    if os.path.exists(record_file) and os.path.getsize(record_file) > 0:
        print("✅ 录音完成")
        return record_file
    else:
        print("❌ 录音失败，请检查麦克风")
        return None

# ===================== 音频转文字 =====================
def audio_to_text(audio_file):
    if not audio_file or not asr_model:
        return ""
    try:
        res = asr_model.generate(input=audio_file, batch_size_s=300)
        if not res or "text" not in res[0]:
            print("⚠️  未识别到内容")
            return ""
        text = res[0]["text"].strip().replace(" ", "")
        print(f"📝 识别内容：{text}")
        # 过滤无效内容
        if not text or len(text) < MIN_VALID_LENGTH:
            print("⚠️  内容过短或无意义")
            return ""
        if set(text).issubset(USELESS_WORDS):
            print("⚠️  识别内容无效")
            return ""
        return text
    except Exception as e:
        print(f"❌ 识别失败：{str(e)}")
        return ""

# ===================== 主程序 =====================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎙️  自动识别版·语音对话助手")
    print("💬 直接输入文字 → 文字对话")
    print("🎤 输入 v → 语音对话")
    print("🚪 输入 q → 退出程序")
    print("="*60)

    while True:
        user_input = input("\n✍️  输入内容（v=语音，q=退出）：").strip()
        if user_input in ["q", "quit", "退出"]:
            print("👋 再见！")
            break
        if user_input in ["v", "voice", "语音"]:
            audio = record_audio()
            text = audio_to_text(audio)
            if not text:
                continue
        else:
            text = user_input
            if not text:
                continue
        # AI回答 + 语音播报
        reply = ask_deepseek(text)
        text_to_speech_play(reply)
