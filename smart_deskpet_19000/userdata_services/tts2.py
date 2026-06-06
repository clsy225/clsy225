import requests
import os
import subprocess
import time
import importlib.util
import re

# ===================== 【你的设备配置，已确认正确】=====================
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
BASE_URL = "https://api.deepseek.com/v1"
PLAY_DEVICE = "plughw:2,0"    # 杰理音响
RECORD_DEVICE = "plughw:2,0"  # 摄像头麦克风
TTS_API_URL = "http://127.0.0.1:9880/"
TEMP_DIR = "/home/linaro/voice_temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# ===================== 无效内容过滤规则 =====================
USELESS_WORDS = {"这", "那", "的", "了", "啊", "嗯", "哦", "呃", "吧", "吗", "呀", "个", "是"}
MIN_VALID_LENGTH = 2

# ===================== 自动检查ASR依赖 =====================
def check_and_install(package, install_name=None):
    if install_name is None:
        install_name = package
    if importlib.util.find_spec(package) is None:
        print(f"📦 正在安装 {package}...")
        subprocess.check_call([
            "python3", "-m", "pip", "install",
            install_name, "-i", "https://mirrors.aliyun.com/pypi/simple/"
        ])

print("🔍 检查ASR依赖...")
check_and_install("funasr")
check_and_install("modelscope")
check_and_install("torchaudio")

from funasr import AutoModel

# ===================== 初始化ASR模型（去掉严格VAD，保证有结果）=====================
print("\n🔄 正在加载语音识别模型...")
try:
    # 去掉vad_model，避免过度过滤
    asr_model = AutoModel(
        model="paraformer-zh", 
        model_revision="v2.0.4"
    )
    print("✅ 语音识别模型加载完成！")
except Exception as e:
    print(f"❌ 模型加载失败：{e}")
    print("💡 已自动切换到纯文字模式")
    asr_model = None

# ===================== 【完全保留你能用的AI对话函数】=====================
def ask_deepseek(prompt):
    print("🤔 正在思考...")
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
                    {"role": "system", "content": "你是一个友好的助手，回答简洁明了，口语化，不超过100字"},
                    {"role": "user", "content": prompt}
                ],
                "stream": False
            }
        )

        if response.status_code == 200:
            result = response.json()
            ai_reply = result["choices"][0]["message"]["content"]
            print(f"🤖 AI回复：{ai_reply}")
            return ai_reply
        else:
            print(f"❌ API错误：{response.status_code}")
            return "API调用失败"

    except Exception as e:
        print(f"❌ 网络错误：{str(e)}")
        return "网络出错"

# ===================== 【完全保留你能用的TTS播报函数】=====================
def text_to_speech_play(text):
    tts_file = f"{TEMP_DIR}/tts_reply.wav"
    if os.path.exists(tts_file):
        os.remove(tts_file)

    data = {
        "text": text,
        "text_language": "zh",
        "cut_punc": "。"
    }

    try:
        print("🔊 正在合成语音（请稍候，本地推理需要时间）...")
        response = requests.post(TTS_API_URL, json=data)
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
            print(f"❌ TTS合成失败，状态码：{response.status_code}")
    except Exception as e:
        print(f"❌ TTS错误：{str(e)}")

# ===================== 录音功能 =====================
def record_audio():
    record_file = f"{TEMP_DIR}/record.wav"
    if os.path.exists(record_file):
        os.remove(record_file)
    
    input("\n🎤 按回车键开始说话，说完再按回车键结束...")
    print("🎙️  正在录音...")
    
    record_process = subprocess.Popen(
        ["arecord", "-D", RECORD_DEVICE, "-f", "S16_LE", "-r", "16000", "-c", "1", record_file],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    
    input("按回车键结束录音...")
    record_process.terminate()
    record_process.wait()
    time.sleep(0.5)
    
    if os.path.exists(record_file) and os.path.getsize(record_file) > 0:
        print("✅ 录音完成！")
        return record_file
    else:
        print("❌ 录音失败，请检查麦克风设备")
        return None

# ===================== 语音转文字（新增空结果判断，彻底解决报错）=====================
def audio_to_text(audio_file):
    if not audio_file or not asr_model:
        return ""
    
    try:
        res = asr_model.generate(
            input=audio_file, 
            batch_size_s=300
        )
        
        # 【核心修复】检查返回结果是否为空
        if not res or len(res) == 0:
            print("⚠️  未识别到任何内容，请重新说话")
            return ""
        
        # 安全获取text字段
        result_item = res[0]
        if "text" not in result_item:
            print("⚠️  识别结果格式异常，请重新说话")
            return ""
        
        text = result_item["text"].strip().replace(" ", "")
        print(f"📝 识别原始内容：{text}")

        # 无效内容过滤
        if not text:
            print("⚠️  未识别到有效人声，请重新说话")
            return ""
        
        text_chars = set(text)
        if text_chars.issubset(USELESS_WORDS):
            print("⚠️  识别内容无意义，已跳过，请重新说话")
            return ""
        
        if len(text) < MIN_VALID_LENGTH:
            print("⚠️  识别内容太短，已跳过，请重新说话")
            return ""
        
        print(f"✅ 有效内容：{text}")
        return text
    
    except Exception as e:
        print(f"❌ 语音识别出错：{str(e)}")
        return ""

# ===================== 主程序（支持语音+文字双模式）=====================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🎙️  修复版语音对话助手")
    print("💡 模式1：直接输入文字，按回车发送")
    print("💡 模式2：输入 v 或 voice，进入语音模式")
    print("🚪 输入 q 退出")
    print("="*60)
    
    while True:
        user_input = input("\n✍️  输入内容（或 v 进入语音模式）：")
        
        if user_input.strip() in ["q", "quit", "退出"]:
            print("👋 再见！")
            break
        
        # 语音模式
        if user_input.strip() in ["v", "voice", "语音"]:
            audio_file = record_audio()
            text = audio_to_text(audio_file)
            if not text.strip():
                continue
        # 文字模式
        else:
            text = user_input.strip()
            if not text:
                continue
        
        # AI对话 + TTS播报
        ai_reply = ask_deepseek(text)
        text_to_speech_play(ai_reply)
