import importlib.util
import json
import os
import re
import shlex
import subprocess
import time
from itertools import count
from pathlib import Path

import requests


DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
BASE_URL = "https://api.deepseek.com/v1"
TTS_API_URL = "http://127.0.0.1:9880/"
PC_TTS_API_URL = "http://192.168.137.1:9880/"
TEMP_DIR = "/home/linaro/voice_temp"
QUEUE_DIR = "/userdata/voice_bridge"
QUEUE_FILE = f"{QUEUE_DIR}/command_queue.json"
RESULT_FILE = f"{QUEUE_DIR}/last_result.json"
LOG_FILE = f"{QUEUE_DIR}/assistant_log.jsonl"

WORKSPACE_ROOT = Path("/home/linaro/.openclaw/workspace").resolve()
USERDATA_ROOT = Path("/userdata").resolve()
ALLOWED_ROOTS = [WORKSPACE_ROOT, USERDATA_ROOT]
COMMAND_IDS = count(int(time.time()))
PENDING_CONFIRMATION = None

SAFE_RUN_PREFIXES = [
    ["python3"],
    ["bash"],
    ["sh"],
    ["xdg-open"],
    ["ls"],
    ["cat"],
    ["sed"],
    ["grep"],
    ["find"],
    ["pwd"],
    ["curl"],
]

USELESS_WORDS = {"这", "那", "的", "了", "啊", "嗯", "哦", "呃", "吧", "吗", "呀", "个", "是"}
MIN_VALID_LENGTH = 2

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(QUEUE_DIR, exist_ok=True)


def append_log(event, payload):
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.time(), "event": event, "payload": payload}, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"log-failed: {exc}")


def is_allowed_path(path_str):
    try:
        candidate = Path(path_str).expanduser().resolve()
    except Exception:
        return False
    return any(candidate == root or root in candidate.parents for root in ALLOWED_ROOTS)


def normalize_path(path_str, default_name=None):
    if not path_str and default_name:
        return str((WORKSPACE_ROOT / default_name).resolve())
    if not path_str:
        return str(WORKSPACE_ROOT)
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = WORKSPACE_ROOT / p
    return str(p.resolve())


def ensure_parent_dir(path_str):
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def get_usb_audio_card():
    try:
        result = subprocess.check_output("aplay -l", shell=True, text=True)
        for line in result.split("\n"):
            if "UACDemoV1.0" in line:
                return line.split("card ")[1].split(":")[0]
    except Exception as exc:
        print(f"audio-card-detect-failed: {exc}")
    return None


AUDIO_CARD = get_usb_audio_card()
if AUDIO_CARD:
    PLAY_DEVICE = f"plughw:{AUDIO_CARD},0"
    RECORD_DEVICE = f"plughw:{AUDIO_CARD},0"
    print(f"audio-card-selected: card{AUDIO_CARD}")
else:
    PLAY_DEVICE = "default"
    RECORD_DEVICE = "default"
    print("audio-card-selected: default")


def check_and_install(package, install_name=None):
    install_name = install_name or package
    if importlib.util.find_spec(package) is None:
        subprocess.check_call(
            [
                "python3",
                "-m",
                "pip",
                "install",
                install_name,
                "-i",
                "https://mirrors.aliyun.com/pypi/simple/",
            ]
        )


check_and_install("funasr")
check_and_install("modelscope")
check_and_install("torchaudio")

from funasr import AutoModel


asr_model = None
try:
    asr_model = AutoModel(model="paraformer-zh", model_revision="v2.0.4")
    print("asr-model-ready")
except Exception as exc:
    print(f"asr-model-failed: {exc}")


def call_deepseek(messages, temperature=0.2):
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        },
        json={
            "model": "deepseek-chat",
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        },
        timeout=90,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def ask_deepseek(prompt):
    try:
        reply = call_deepseek(
            [
                {
                    "role": "system",
                    "content": "You are a concise assistant. Reply in colloquial Chinese in under 120 characters.",
                },
                {"role": "user", "content": prompt},
            ]
        )
        print(f"assistant-reply: {reply}")
        return reply
    except Exception as exc:
        print(f"deepseek-chat-failed: {exc}")
        return "AI 请求失败。"


def plan_pc_action(text):
    system_prompt = (
        "You are a Linux voice command router for an embedded board assistant. "
        "Return strict JSON only, without markdown. "
        "Allowed actions are chat, open_browser, play_video, open_file, open_dir, run_command, write_code, fetch_info, read_file. "
        "Use payload fields as needed: url, path, command, query, filename, content, description. "
        "All paths must target Linux and default under /home/linaro/.openclaw/workspace when not provided. "
        "For write_code, create code content directly in payload.content and target path in payload.path. "
        "For fetch_info, include payload.query or payload.url. "
        "For risky actions like run_command or write_code, reply should briefly say what will happen. "
        "Schema: {\"action\":\"...\",\"reply\":\"...\",\"payload\":{...}}"
    )
    try:
        raw = call_deepseek(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
        )
        match = re.search(r"\{.*\}", raw.strip(), re.S)
        cleaned = match.group(0) if match else raw.strip()
        data = json.loads(cleaned)
        data.setdefault("payload", {})
        data.setdefault("reply", "")
        if data.get("action") not in {
            "chat",
            "open_browser",
            "play_video",
            "open_file",
            "open_dir",
            "run_command",
            "write_code",
            "fetch_info",
            "read_file",
        }:
            raise ValueError(f"unsupported action: {data.get('action')}")
        return data
    except Exception as exc:
        print(f"route-plan-failed: {exc}")
        return {"action": "chat", "reply": ask_deepseek(text), "payload": {}}


def _try_tts_http(base_url, text, tts_file):
    endpoints = [base_url.rstrip('/'), base_url.rstrip('/') + '/tts']
    payloads = [
        {"text": text, "text_language": "zh", "cut_punc": "。"},
        {"text": text, "text_lang": "zh"},
        {"text": text},
    ]
    last_error = None
    for endpoint in endpoints:
        for payload in payloads:
            try:
                response = requests.post(endpoint, json=payload, timeout=60)
                ctype = response.headers.get('content-type', '')
                if response.status_code == 200 and ('audio' in ctype or 'application/octet-stream' in ctype or len(response.content) > 1024):
                    with open(tts_file, 'wb') as f:
                        f.write(response.content)
                    subprocess.run(["aplay", "-D", PLAY_DEVICE, tts_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    print(f"tts-played-via-http: {endpoint}")
                    return True
                last_error = f"{endpoint} status={response.status_code} ctype={ctype}"
            except Exception as exc:
                last_error = str(exc)
    print(f"tts-http-attempt-failed: {last_error}")
    return False


def text_to_speech_play(text):
    if not text:
        return
    tts_file = os.path.join(TEMP_DIR, "tts_reply.wav")
    if os.path.exists(tts_file):
        os.remove(tts_file)

    # 方案1：优先电脑上的 GPT-SoVITS API
    if _try_tts_http(PC_TTS_API_URL, text, tts_file):
        return

    # 方案2：开发板本机 HTTP TTS 服务
    if _try_tts_http(TTS_API_URL, text, tts_file):
        return

    # 方案3：回退到你原来的 GPT-SoVITS 脚本
    fallback_script = "/home/linaro/tts_speak.sh"
    if os.path.exists(fallback_script):
        try:
            print("tts-fallback-script-start")
            subprocess.run([fallback_script, text], check=True, timeout=180)
            print("tts-played-via-script")
            return
        except Exception as exc:
            print(f"tts-fallback-failed: {exc}")

    print("tts-unavailable")


def record_audio():
    record_file = os.path.join(TEMP_DIR, "record.wav")
    if os.path.exists(record_file):
        os.remove(record_file)
    input("press-enter-to-start-recording")
    proc = subprocess.Popen(
        ["arecord", "-D", RECORD_DEVICE, "-f", "S16_LE", "-r", "16000", "-c", "1", record_file],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    input("press-enter-to-stop-recording")
    proc.terminate()
    proc.wait()
    time.sleep(0.5)
    if os.path.exists(record_file) and os.path.getsize(record_file) > 0:
        print("recording-finished")
        return record_file
    print("recording-failed")
    return None


def audio_to_text(audio_file):
    if not audio_file or not asr_model:
        return ""
    try:
        res = asr_model.generate(input=audio_file, batch_size_s=300)
        if not res or "text" not in res[0]:
            print("asr-empty")
            return ""
        text = res[0]["text"].strip().replace(" ", "")
        print(f"recognized-text: {text}")
        if not text or len(text) < MIN_VALID_LENGTH:
            return ""
        if set(text).issubset(USELESS_WORDS):
            return ""
        return text
    except Exception as exc:
        print(f"asr-failed: {exc}")
        return ""


def read_json_file(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json_file(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def enqueue_pc_command(action, payload, original_text):
    queue = read_json_file(QUEUE_FILE, [])
    command = {
        "id": next(COMMAND_IDS),
        "action": action,
        "payload": payload,
        "text": original_text,
        "created_at": time.time(),
    }
    queue.append(command)
    write_json_file(QUEUE_FILE, queue)
    return command


def wait_for_result(command_id, timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        result = read_json_file(RESULT_FILE, {})
        if result.get("command_id") == command_id:
            return result
        time.sleep(1)
    return {"ok": False, "status": "timeout", "message": "PC agent timeout."}


def needs_confirmation(action):
    return action in {"run_command", "write_code"}


def is_confirmation_text(text):
    normalized = text.strip().lower()
    return normalized in {"确认", "确定", "继续", "执行", "yes", "ok", "好的"}


def run_shell_command(command_text):
    argv = shlex.split(command_text)
    if not argv:
        return False, "空命令"
    if not any(argv[: len(prefix)] == prefix for prefix in SAFE_RUN_PREFIXES):
        return False, "命令不在白名单内"
    cwd = str(WORKSPACE_ROOT)
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=120, cwd=cwd)
    output = (proc.stdout or "") + (proc.stderr or "")
    output = output.strip()[:2000]
    return proc.returncode == 0, output or "命令执行完成"


def open_with_xdg(target):
    try:
        subprocess.Popen(["xdg-open", target], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True, f"已打开 {target}"
    except Exception as exc:
        return False, f"打开失败: {exc}"


def handle_fetch_info(payload):
    url = payload.get("url")
    query = payload.get("query")
    try:
        if url:
            resp = requests.get(url, timeout=20)
            text = resp.text[:1500]
            return True, f"已获取页面内容，前面内容是：{text}"
        if query:
            answer = ask_deepseek(f"请简短回答并尽量基于常识：{query}")
            return True, answer
        return False, "没有提供查询内容"
    except Exception as exc:
        return False, f"获取信息失败: {exc}"


def handle_write_code(payload):
    path = normalize_path(payload.get("path"), payload.get("filename") or "generated_code.py")
    if not is_allowed_path(path):
        return False, "路径不允许"
    content = payload.get("content", "").strip()
    description = payload.get("description", "").strip()
    if not content and description:
        content = call_deepseek(
            [
                {"role": "system", "content": "You write concise working code. Output code only."},
                {"role": "user", "content": description},
            ],
            temperature=0.2,
        )
    if not content:
        return False, "没有可写入的代码内容"
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return True, f"代码已写入 {path}"


def handle_read_file(payload):
    path = normalize_path(payload.get("path"))
    if not is_allowed_path(path) or not os.path.exists(path):
        return False, "文件不存在或路径不允许"
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        data = f.read(1500)
    return True, f"文件前面内容是：{data}"


def execute_plan(plan, original_text):
    action = plan.get("action", "chat")
    payload = plan.get("payload") or {}
    append_log("plan", {"text": original_text, "plan": plan})

    if action == "chat":
        reply = plan.get("reply") or ask_deepseek(original_text)
        return True, reply

    if action == "open_browser":
        url = payload.get("url") or "https://www.baidu.com"
        return open_with_xdg(url)

    if action == "play_video":
        url = payload.get("url") or "https://www.bilibili.com"
        return open_with_xdg(url)

    if action == "open_file":
        path = normalize_path(payload.get("path"))
        if not is_allowed_path(path):
            return False, "路径不允许"
        return open_with_xdg(path)

    if action == "open_dir":
        path = normalize_path(payload.get("path"))
        if not is_allowed_path(path):
            return False, "路径不允许"
        return open_with_xdg(path)

    if action == "run_command":
        cmd = payload.get("command", "")
        return run_shell_command(cmd)

    if action == "write_code":
        return handle_write_code(payload)

    if action == "fetch_info":
        return handle_fetch_info(payload)

    if action == "read_file":
        return handle_read_file(payload)

    return False, "暂不支持这个动作"


def handle_user_text(text):
    global PENDING_CONFIRMATION

    if PENDING_CONFIRMATION and is_confirmation_text(text):
        plan = PENDING_CONFIRMATION
        PENDING_CONFIRMATION = None
        ok, final_reply = execute_plan(plan, "confirmed-action")
        print(f"confirmed-result: {final_reply}")
        text_to_speech_play(final_reply)
        return

    plan = plan_pc_action(text)
    action = plan.get("action", "chat")

    if needs_confirmation(action):
        PENDING_CONFIRMATION = plan
        wait_reply = plan.get("reply") or "这个操作需要确认，请说确认继续执行。"
        print(f"need-confirmation: {wait_reply}")
        text_to_speech_play(wait_reply)
        return

    ok, reply = execute_plan(plan, text)
    print(f"assistant-result: {reply}")
    text_to_speech_play(reply)


def main():
    print("=" * 60)
    print("tts4 voice assistant enhanced")
    print("type text to send directly")
    print("type v for voice input")
    print("type q to quit")
    print("supports browser, files, code writing, safe commands, info fetch")
    print("=" * 60)
    while True:
        user_input = input("input(v=voice,q=quit): ").strip()
        if user_input in {"q", "quit", "exit"}:
            break
        if user_input in {"v", "voice"}:
            audio = record_audio()
            text = audio_to_text(audio)
            if not text:
                continue
        else:
            text = user_input
            if not text:
                continue
        handle_user_text(text)


if __name__ == "__main__":
    main()
