#!/usr/bin/env python3
import json
import html
import os
import re
import signal
import shlex
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urlencode

import functools
import requests
from flask import Flask, Response, jsonify, request

BASE = Path('/userdata')

# AI 引擎
import sys as _sys
_sys.path.insert(0, str(BASE / 'openclaw_workspace'))
from ai_engine import AIEngine as _AIEngine
ai_engine = _AIEngine()
ai_state = {
    'mode': ai_engine.config.get('mode', 'local'),
    'requests': 0,
    'total_local': 0,
    'total_cloud': 0,
    'avg_time_local': 0,
    'avg_time_cloud': 0,
    'last_answer': '',
    'preset_questions': [
        '你是谁？',
        '现在几点了？',
        '帮我介绍一下RK3588',
        '写一首关于桌宠的诗',
        'NPU是什么？',
        '1+1等于几？',
    ]
}

LIVE2D_URL = '/proxy/live2d'
MOUSE_URL = '/proxy/mouse/'
SPEAKER_URL = '/proxy/speaker/'
LIVE2D_SERVER = BASE / 'openclaw_workspace/live2d_gesture_merged.py'
TTS4_PATH = BASE / 'tts4.py'
TTS5_WEB_PATH = BASE / 'tts5_web.py'
GPT_SOVITS_DIR = Path('/userdata/rknn_voice_test')
GPT_SOVITS_CMD = [
    'python', 'api_rknn_keepalive.py',
    '-g', '/home/linaro/GPT_weights_v2Pro/ldnn-e15.ckpt',
    '-s', '/home/linaro/SoVITS_weights_v2Pro/xxx_e8_s640.pth',
    '-a', '0.0.0.0',
    '-p', '9880',
    '-dr', '/home/linaro/witch.wav',
    '-dt', '魔女',
    '-dl', 'ja',
]
PORT = 19000

app = Flask(__name__)
state = {
    'status': 'ready',
    'live2d': False,
    'mouse': False,
    'last_action': '',
    'last_text': '',
    'active_mode': 'idle',
    'active_title': 'Live2D 待机中',
    'tts': {
        'status': 'idle',
        'text': '',
        'started_at': None,
        'ended_at': None,
        'code': None,
        'error': '',
    },
    'pet': {
        'auto_mode': 'unknown',
        'last_motion': '',
        'resume_in_sec': None,
        'speaking': False,
    },
}

TTS_PROC = None
VISUAL_PROC = None
ASR_MODEL = None
MIN_VALID_LENGTH = 2
USELESS_WORDS = {"这", "那", "的", "了", "啊", "嗯", "哦", "呃", "吧", "吗", "呀", "个", "是"}
VISION_PATTERNS = 'media4.py|camera44.py|renlian15.py|zongtitrack4.py|mediapipe_mouse_control.py|emotion_camera.py|tts3.py'
PORT_NAMES = {
    5000: '手势识别',
    5001: '鼠标控制',
    5002: 'Live2D 桌宠',
    8080: '摄像头 RKNN 检测',
    8083: '绫地宁宁',
    8088: '总跟踪',
    8095: '声纹识别',
    9880: 'GPT-SoVITS',
}

WORKSPACE_ROOT = Path('/home/linaro/.openclaw/workspace').resolve()
USERDATA_ROOT = Path('/userdata').resolve()
ALLOWED_ROOTS = [WORKSPACE_ROOT, USERDATA_ROOT]
SAFE_RUN_PREFIXES = [
    ['python3'],
    ['bash'],
    ['sh'],
    ['xdg-open'],
    ['ls'],
    ['cat'],
    ['sed'],
    ['grep'],
    ['find'],
    ['pwd'],
    ['curl'],
]
PENDING_VOICE_COMMAND = None


def load_asr_model():
    global ASR_MODEL
    if ASR_MODEL is not None:
        return ASR_MODEL
    try:
        from funasr import AutoModel
        ASR_MODEL = AutoModel(model='paraformer-zh', model_revision='v2.0.4')
        return ASR_MODEL
    except Exception:
        ASR_MODEL = False
        return False


def detect_usb_play_device():
    try:
        result = subprocess.check_output('aplay -l', shell=True, text=True)
        for line in result.splitlines():
            if 'UACDemoV1.0' in line:
                return 'plughw:' + line.split('card ')[1].split(':')[0] + ',0'
    except Exception:
        pass
    return 'default'


def detect_usb_record_device():
    preferred = 'plughw:CARD=UACDemoV10,DEV=0'
    try:
        result = subprocess.check_output('arecord -L', shell=True, text=True)
        if preferred in result:
            return preferred
        for key in ['hw:CARD=UACDemoV10,DEV=0', 'plughw:CARD=Camera,DEV=0', 'hw:CARD=Camera,DEV=0', 'default']:
            if key in result:
                return key
    except Exception:
        pass
    return preferred


def is_allowed_path(path_str):
    try:
        candidate = Path(path_str).expanduser().resolve()
    except Exception:
        return False
    return any(candidate == root or root in candidate.parents for root in ALLOWED_ROOTS)


def normalize_workspace_path(path_str, default_name=None):
    if not path_str and default_name:
        return str((WORKSPACE_ROOT / default_name).resolve())
    if not path_str:
        return str(WORKSPACE_ROOT)
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return str(path.resolve())


def open_with_xdg(target):
    try:
        subprocess.Popen(['xdg-open', target], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True, f'已打开 {target}'
    except Exception as exc:
        return False, f'打开失败: {exc}'


def run_safe_shell_command(command_text):
    try:
        argv = shlex.split(command_text)
    except Exception as exc:
        return False, f'命令解析失败: {exc}'
    if not argv:
        return False, '空命令'
    if not any(argv[:len(prefix)] == prefix for prefix in SAFE_RUN_PREFIXES):
        return False, '命令不在白名单内'
    proc = subprocess.run(argv, cwd=str(WORKSPACE_ROOT), capture_output=True, text=True, timeout=120)
    output = ((proc.stdout or '') + (proc.stderr or '')).strip()[:3000]
    return proc.returncode == 0, output or '命令执行完成'


def extract_code_content(text):
    text = (text or '').strip()
    if not text:
        return ''
    match = re.search(r'```[a-zA-Z0-9_+-]*\s*(.*?)\s*```', text, re.S)
    if match:
        return match.group(1).strip()
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith('```'):
            break
        if stripped.startswith(('这个', '这个简单', '以上', '说明', '解释')):
            break
        lines.append(line)
    return '\n'.join(lines).strip()


def write_code_file(payload):
    path = normalize_workspace_path(payload.get('path') or payload.get('filename'), payload.get('filename') or 'voice_generated.py')
    if not is_allowed_path(path):
        return False, '路径不允许'
    content = (payload.get('content') or '').strip()
    description = (payload.get('description') or '').strip()
    if not content and description:
        result = run_ai_prompt('请只输出代码，不要解释：' + description)
        content = extract_code_content(result.get('answer') or '')
    if not content:
        return False, '没有可写入的代码内容'
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(content, encoding='utf-8')
    return True, f'代码已写入 {path}'


def plan_voice_command(text):
    text = (text or '').strip()
    lowered = text.lower()
    if not text:
        return {'action': 'chat', 'reply': '没有识别到语音内容', 'payload': {}}
    if any(k in text for k in ['打开百度', '百度一下', '上百度', '百度搜索']) or 'baidu' in lowered:
        return {'action': 'open_browser', 'reply': '打开百度', 'payload': {'url': 'https://www.baidu.com'}}
    if any(k in text for k in ['打开浏览器', '打开网页']):
        return {'action': 'open_browser', 'reply': '打开浏览器', 'payload': {'url': 'https://www.baidu.com'}}
    if any(k in text for k in ['写代码', '生成代码', '创建代码']):
        filename = 'voice_generated.py'
        match = re.search(r'([A-Za-z0-9_\-./]+\.py)', text)
        if match:
            filename = match.group(1)
        return {'action': 'write_code', 'reply': '准备写代码，需要确认后执行', 'payload': {'filename': filename, 'description': text}}
    if text.startswith('运行命令') or text.startswith('执行命令'):
        command = text.replace('运行命令', '', 1).replace('执行命令', '', 1).strip()
        return {'action': 'run_command', 'reply': '准备运行命令，需要确认后执行', 'payload': {'command': command}}
    result = run_ai_prompt(
        '把这句话规划成严格JSON，不要markdown。允许action: chat, open_browser, write_code, run_command, read_file。'
        'payload可包含url,path,filename,content,description,command。'
        '如果是打开百度用open_browser url=https://www.baidu.com；如果是写代码用write_code并给description。'
        '原话：' + text
    )
    raw = (result.get('answer') or '').strip()
    try:
        match = re.search(r'\{.*\}', raw, re.S)
        data = json.loads(match.group(0) if match else raw)
        if data.get('action') in {'chat', 'open_browser', 'write_code', 'run_command', 'read_file'}:
            data.setdefault('payload', {})
            data.setdefault('reply', '')
            return data
    except Exception:
        pass
    return {'action': 'chat', 'reply': raw or ('收到语音：' + text), 'payload': {}}


def execute_voice_plan(plan):
    action_name = plan.get('action', 'chat')
    payload = plan.get('payload') or {}
    if action_name == 'chat':
        return True, plan.get('reply') or '收到'
    if action_name == 'open_browser':
        return open_with_xdg(payload.get('url') or 'https://www.baidu.com')
    if action_name == 'write_code':
        return write_code_file(payload)
    if action_name == 'run_command':
        return run_safe_shell_command(payload.get('command') or '')
    if action_name == 'read_file':
        path = normalize_workspace_path(payload.get('path'))
        if not is_allowed_path(path) or not os.path.exists(path):
            return False, '文件不存在或路径不允许'
        return True, Path(path).read_text(encoding='utf-8', errors='ignore')[:3000]
    return False, '暂不支持这个动作'


def voice_plan_needs_confirmation(plan):
    return plan.get('action') in {'write_code', 'run_command'}


def ensure_gpt_sovits():
    try:
        out = subprocess.check_output("pgrep -af 'api_rknn_keepalive.py.* -p 9880|api.py.* -p 9880'", shell=True, text=True, stderr=subprocess.DEVNULL)
        if out.strip():
            return True
    except Exception:
        pass
    if wait_port(9880, timeout=1.0):
        return True
    try:
        subprocess.Popen(
            GPT_SOVITS_CMD,
            cwd=str(GPT_SOVITS_DIR),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )
    except Exception:
        return False
    return wait_port(9880, timeout=25.0)


def play_tts_http(text):
    tts_file = '/tmp/dashboard_tts_reply.wav'
    endpoints = ['http://127.0.0.1:9880', 'http://127.0.0.1:9880/tts', 'http://192.168.137.1:9880', 'http://192.168.137.1:9880/tts']
    payloads = [
        {'text': text, 'text_language': 'zh', 'cut_punc': '。'},
        {'text': text, 'text_lang': 'zh'},
        {'text': text},
    ]
    last_error = ''
    for endpoint in endpoints:
        for payload in payloads:
            try:
                r = requests.post(endpoint, json=payload, timeout=90)
                ctype = r.headers.get('Content-Type', '')
                if r.status_code == 200 and ('audio' in ctype or 'application/octet-stream' in ctype or len(r.content) > 1024):
                    with open(tts_file, 'wb') as f:
                        f.write(r.content)
                    device = detect_usb_play_device()
                    proc = subprocess.Popen(['aplay', '-D', device, tts_file], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
                    return proc, {'endpoint': endpoint, 'payload': payload, 'device': device}
                last_error = f'{endpoint} status={r.status_code} ctype={ctype} body={r.text[:200]}'
            except Exception as exc:
                last_error = f'{endpoint} error={exc}'
    raise RuntimeError(last_error or 'tts http unavailable')


def speak_reply_async(text):
    text = (text or '').strip()
    if not text:
        return

    def worker():
        global TTS_PROC
        if TTS_PROC and TTS_PROC.poll() is None:
            return
        try:
            proc, meta = play_tts_http(text[:180])
            TTS_PROC = proc
            state['tts'] = {'status': 'running', 'text': text[:180], 'started_at': time.time(), 'ended_at': None, 'code': None, 'error': '', 'meta': meta}
            state['pet']['speaking'] = True
            state['last_action'] = 'voice_command_tts'
        except Exception as exc:
            state['tts'] = {'status': 'error', 'text': text[:180], 'started_at': None, 'ended_at': None, 'code': None, 'error': str(exc)}
            state['pet']['speaking'] = False

    threading.Thread(target=worker, daemon=True).start()


def start_if_missing(pattern, cmd, env=None):
    try:
        subprocess.run(
            f"pgrep -af '{pattern}'",
            shell=True,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return True
    except Exception:
        subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
            env=env,
        )
        return False


def wait_port(port, timeout=12.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f'http://127.0.0.1:{port}/', timeout=1.5)
            if r.status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(0.25)
    return False


def ensure_all():
    os.environ.setdefault('DISPLAY', ':0')
    os.environ.setdefault('XAUTHORITY', '/home/linaro/.Xauthority')
    os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')
    os.environ.setdefault('QT_OPENGL', 'software')
    os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')
    if not wait_port(5002, timeout=0.8):
        subprocess.Popen(
            ['python3', str(LIVE2D_SERVER)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
            env={**os.environ, 'DISABLE_CAMERA_ON_START': '1'},
        )
    if not wait_port(8095, timeout=0.8):
        subprocess.Popen(
            ['python3', str(TTS5_WEB_PATH)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )
    ensure_gpt_sovits()
    state['live2d'] = wait_port(5002, timeout=6.0)


def live2d_camera_request(action, timeout=8):
    try:
        ensure_all()
        r = requests.post(f'http://127.0.0.1:5002/api/camera/{action}', timeout=timeout)
        return r.status_code < 500
    except Exception:
        return False


def start_live2d_gesture_camera():
    return live2d_camera_request('start', timeout=20)


def stop_live2d_gesture_camera():
    return live2d_camera_request('stop', timeout=8)


def ensure_stack_dependencies():
    ensure_all()
    deps = {
        'live2d': wait_port(5002, timeout=2.0),
        'speaker': wait_port(8095, timeout=2.0),
        'gpt_sovits': wait_port(9880, timeout=2.0),
        'mouse_aux': wait_port(5001, timeout=2.0),
        'track': wait_port(8088, timeout=2.0),
    }
    return deps


def proxy_headers(upstream_headers):
    allowed = {}
    for key in ['Content-Type', 'Cache-Control', 'ETag', 'Last-Modified']:
        if key in upstream_headers:
            allowed[key] = upstream_headers[key]
    return allowed


def rewrite_live2d_html(html):
    prefix = '/proxy/live2d'
    html = (
        html.replace('src="/runtime/', f'src="{prefix}/runtime/')
        .replace("src='/runtime/", f"src='{prefix}/runtime/")
        .replace('href="/runtime/', f'href="{prefix}/runtime/')
        .replace("href='/runtime/", f"href='{prefix}/runtime/")
        .replace("const MODEL_URL = '/model/' + encodeURIComponent('", f"const MODEL_URL = '{prefix}/model/' + encodeURIComponent('")
        .replace("window.open('/?mode=pet'", f"window.open('{prefix}?mode=pet'")
        .replace("window.open('/?mode=debug'", f"window.open('{prefix}?mode=debug'")
        .replace('src="/video_feed', f'src="{prefix}/video_feed')
        .replace("src='/video_feed", f"src='{prefix}/video_feed")
        .replace('fetch("/video_feed', f'fetch("{prefix}/video_feed')
        .replace("fetch('/video_feed", f"fetch('{prefix}/video_feed")
        .replace('fetch("/api/', f'fetch("{prefix}/api/')
        .replace("fetch('/api/", f"fetch('{prefix}/api/")
        .replace('action="/api/', f'action="{prefix}/api/')
        .replace("action='/api/", f"action='{prefix}/api/")
        .replace('href="/control', f'href="{prefix}/control')
        .replace("href='/control", f"href='{prefix}/control")
        .replace('href="/debug-actions', f'href="{prefix}/debug-actions')
        .replace("href='/debug-actions", f"href='{prefix}/debug-actions")
    )

    inject = '''
<script>
(function(){
  const state = { auto: 'unknown', lastMotion: '', resumeInSec: null, speaking: false };
  let resumeTimer = null;

  function notify() {
    try {
      if (window.parent && window.parent !== window && typeof window.parent.postMessage === 'function') {
        window.parent.postMessage({ source: 'live2d-proxy', type: 'pet-status', payload: state }, '*');
      }
    } catch (e) {}
  }

  function patchSetStatus(){
    if (typeof window.setStatus !== 'function') return;
    const orig = window.setStatus;
    window.setStatus = function(text){
      try {
        const s = String(text || '');
        if (s.includes('已开启自动切换')) state.auto = 'on';
        if (s.includes('已暂停自动切换') || s.includes('自动切换已暂停')) state.auto = 'paused';
        const m = s.match(/已播放动作\s+(.+)/);
        if (m) state.lastMotion = m[1];
        if (s.includes('挥手模式')) state.lastMotion = '挥手';
        if (s.includes('比心模式')) state.lastMotion = '比心';
        notify();
      } catch (e) {}
      return orig.apply(this, arguments);
    }
  }

  function wrapFn(name, before, after){
    if (typeof window[name] !== 'function') return;
    const orig = window[name];
    window[name] = function(){
      try { before && before.apply(this, arguments); } catch (e) {}
      const ret = orig.apply(this, arguments);
      try {
        if (ret && typeof ret.then === 'function') {
          return ret.finally(() => { try { after && after.apply(this, arguments); } catch (e) {} });
        }
      } catch (e) {}
      try { after && after.apply(this, arguments); } catch (e) {}
      return ret;
    }
  }

  function attach(){
    patchSetStatus();
    wrapFn('startAutoSwitch', function(){ state.auto = 'on'; state.resumeInSec = null; if (resumeTimer) { clearInterval(resumeTimer); resumeTimer = null; } notify(); });
    wrapFn('stopAutoSwitch', function(){ state.auto = 'paused'; notify(); });
    wrapFn('resumeAutoSwitchLater', function(ms){
      const total = Math.max(1, Math.ceil((Number(ms)||15000)/1000));
      state.resumeInSec = total;
      notify();
      if (resumeTimer) clearInterval(resumeTimer);
      let left = total;
      resumeTimer = setInterval(() => {
        left -= 1;
        state.resumeInSec = left > 0 ? left : null;
        notify();
        if (left <= 0) {
          clearInterval(resumeTimer);
          resumeTimer = null;
        }
      }, 1000);
    });
    wrapFn('triggerWaveMode', function(){ state.lastMotion = '挥手'; notify(); });
    wrapFn('triggerHeartMode', function(){ state.lastMotion = '比心'; notify(); });

    window.addEventListener('message', async (event) => {
      const data = event && event.data;
      if (!data || data.source !== 'dashboard-19000') return;
      try {
        if (data.type === 'pet-action') {
          const action = data.action;
          if (action === 'auto-on' && typeof window.startAutoSwitch === 'function') window.startAutoSwitch();
          if (action === 'auto-off' && typeof window.stopAutoSwitch === 'function') window.stopAutoSwitch('已从控制台暂停自动切换');
          if (action === 'wave' && typeof window.triggerWaveMode === 'function') await window.triggerWaveMode();
          if (action === 'heart' && typeof window.triggerHeartMode === 'function') await window.triggerHeartMode();
          if (action === 'idle' && typeof window.playIdle === 'function') await window.playIdle();
        }
        if (data.type === 'tts-state') {
          state.speaking = !!(data.payload && data.payload.speaking);
          notify();
          if (state.speaking && typeof window.stopAutoSwitch === 'function') {
            window.stopAutoSwitch('TTS 播放中，临时暂停自动切换');
          } else if (!state.speaking && typeof window.resumeAutoSwitchLater === 'function') {
            window.resumeAutoSwitchLater(8000);
          }
        }
      } catch (e) {}
    });

    notify();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', attach);
  } else {
    attach();
  }
})();
</script>
'''
    return html.replace('</body>', inject + '\n</body>') if '</body>' in html else html + inject


def rewrite_generic_html(html, replacements):
    for old, new in replacements:
        html = html.replace(old, new)
    return html


def rewrite_mediapipe_mouse_html(html):
    return rewrite_generic_html(html, [
        ('src="/video_feed"', 'src="/proxy/mouse_aux/video_feed"'),
        ("src='/video_feed'", "src='/proxy/mouse_aux/video_feed'"),
        ('fetch("/config"', 'fetch("/proxy/mouse_aux/config"'),
        ("fetch('/config'", "fetch('/proxy/mouse_aux/config'"),
    ])


def speaker_proxy_request(method, subpath=''):
    target = f'http://127.0.0.1:8095/{subpath}' if subpath else 'http://127.0.0.1:8095/'
    headers = {k: v for k, v in request.headers if k.lower() not in {'host', 'content-length'}}
    if method == 'GET':
        r = requests.get(target, params=request.args, headers=headers, timeout=180)
    else:
        r = requests.request(method, target, params=request.args, headers=headers, data=request.get_data(), timeout=180)
    ctype = r.headers.get('Content-Type', '')
    if 'text/html' in ctype and method == 'GET':
        html = r.text
        prefix = '/proxy/speaker'
        replacements = [
            ('src="/', 'src="' + prefix + '/'),
            ("src='/", "src='" + prefix + '/'),
            ('href="/', 'href="' + prefix + '/'),
            ("href='/", "href='" + prefix + '/'),
            ('fetch("/', 'fetch("' + prefix + '/'),
            ("fetch('/", "fetch('" + prefix + '/'),
            ('jsonFetch("/', 'jsonFetch("' + prefix + '/'),
            ("jsonFetch('/", "jsonFetch('" + prefix + '/'),
            ('action="/', 'action="' + prefix + '/'),
            ("action='/", "action='" + prefix + '/'),
        ]
        for old, new in replacements:
            html = html.replace(old, new)
        rheaders = proxy_headers(r.headers)
        rheaders['Content-Type'] = rheaders.get('Content-Type', 'text/html; charset=utf-8')
        return html, r.status_code, rheaders
    return Response(r.content, status=r.status_code, headers=proxy_headers(r.headers))


def page():
    return '''<!doctype html>
<html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>智能桌宠交互中心</title>
<style>
:root{--bg:#0b1020;--card:#121a2f;--card2:#18233d;--line:rgba(255,255,255,.08);--text:#e9eefc;--muted:#9fb0d4;--blue:#5b8cff;--blue2:#7aa2ff;--green:#19c37d;--red:#e05d5d;--amber:#e7b95b}
*{box-sizing:border-box}body{margin:0;font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:radial-gradient(circle at top,#17233f 0,#0b1020 48%,#080c18 100%);color:var(--text)}
.top{padding:28px 18px 18px}.hero{max-width:1480px;margin:0 auto;display:flex;gap:18px;align-items:stretch}.hero-copy,.hero-stats{background:linear-gradient(180deg,rgba(255,255,255,.05),rgba(255,255,255,.02));border:1px solid var(--line);border-radius:24px;padding:22px}.hero-copy{flex:1.25}.hero-copy h1{margin:0 0 10px;font-size:30px}.sub{color:var(--muted);line-height:1.65;font-size:14px}.hero-stats{width:320px;display:flex;flex-direction:column;justify-content:center;gap:12px}.hero-pill{padding:12px 14px;border-radius:16px;background:#0f1728;border:1px solid rgba(255,255,255,.06);font-size:14px}
.grid{max-width:1480px;margin:0 auto;padding:0 18px 22px;display:grid;grid-template-columns:minmax(420px,1.15fr) minmax(320px,.85fr);gap:18px}.card{background:linear-gradient(180deg,var(--card),var(--card2));border:1px solid var(--line);border-radius:24px;overflow:hidden;box-shadow:0 16px 50px rgba(0,0,0,.25)}
.card h2{display:flex;justify-content:space-between;align-items:center;padding:18px 18px 0;margin:0;font-size:18px}.card h2 small{color:var(--muted);font-weight:500;font-size:12px}
iframe{width:100%;height:520px;border:0;background:#fff}.live2d-frame{background:linear-gradient(180deg,#0f1728,#10192d)}
.pane{padding:16px}.section{padding:14px 0;border-top:1px solid rgba(255,255,255,.06)}.section:first-child{padding-top:0;border-top:none}.section-title{margin:0 0 10px;font-size:13px;color:#c7d5f4;text-transform:uppercase;letter-spacing:.8px}
input,textarea,button,select{width:100%;border-radius:14px;border:1px solid rgba(255,255,255,.10);background:#0c1425;color:#fff;padding:12px 13px;font-size:14px;transition:.18s ease}
input:focus,textarea:focus,select:focus{outline:none;border-color:rgba(122,162,255,.65);box-shadow:0 0 0 4px rgba(91,140,255,.12)}
button{background:linear-gradient(180deg,var(--blue2),var(--blue));border:none;margin-top:8px;cursor:pointer;font-weight:700;box-shadow:0 10px 24px rgba(91,140,255,.28)} button:hover{transform:translateY(-1px);filter:brightness(1.03)} button.warn{background:linear-gradient(180deg,#f17373,var(--red));box-shadow:0 10px 24px rgba(224,93,93,.22)} button.good{background:linear-gradient(180deg,#34d399,var(--green));box-shadow:0 10px 24px rgba(33,201,135,.22)} button.secondary{background:linear-gradient(180deg,#6782b8,#506890);box-shadow:none}
.row{display:grid;grid-template-columns:1fr 1fr;gap:10px}pre,textarea[readonly]{white-space:pre-wrap;background:#0a1220;padding:12px;border-radius:14px;min-height:110px;border:1px solid rgba(255,255,255,.06)}
.small{font-size:12px;color:var(--muted);line-height:1.5}.small.tip{margin-top:8px;padding:10px 12px;border-radius:12px;background:rgba(91,140,255,.08);border:1px solid rgba(91,140,255,.14)}
.badges{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0 4px}.badge{padding:8px 11px;border-radius:999px;background:#21314f;font-size:12px;border:1px solid rgba(255,255,255,.05)}.badge.on{background:rgba(33,201,135,.18);color:#b6f7dd}.badge.off{background:rgba(224,93,93,.16);color:#ffd1d1}.badge.wait{background:rgba(231,185,91,.16);color:#ffe6b5}
.viewer-tabs{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:10px}.viewer-tabs button{margin-top:0;padding:10px 8px}.panel-note{padding:12px 14px;margin-top:12px;border-radius:14px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.06);color:var(--muted);font-size:13px;line-height:1.55}
.wide-card{grid-column:1 / span 2}.wide-card iframe{height:600px}.speaker-card{grid-column:1 / span 2}.speaker-card iframe{height:1020px}footer{max-width:1480px;margin:0 auto;padding:0 18px 20px;color:var(--muted);font-size:12px}
@media (max-width: 1180px){.grid{grid-template-columns:1fr}.wide-card,.speaker-card{grid-column:1 / span 1}.wide-card iframe{height:520px}.speaker-card iframe{height:1180px}.hero{flex-direction:column}}
@media (max-width: 720px){.top{padding:20px 14px 16px}.grid{padding:14px}.row,.viewer-tabs{grid-template-columns:1fr}.hero-copy h1{font-size:24px}.hero-pill{width:100%}}
</style></head><body>
<div class="top"><div class="hero"><div class="hero-copy"><h1>智能桌宠交互中心</h1><div class="sub">把 Live2D 桌宠、视觉功能切换、页面语音和声纹建库收在一个更顺手的控制台里。上面看桌宠，右侧发动作和语音，下面切视觉页与做声纹识别。</div></div><div class="hero-stats"><div class="hero-pill">🎭 Live2D 桌宠</div><div class="hero-pill">🎤 页面 TTS</div><div class="hero-pill">🧠 声纹建库 / 识别</div></div></div></div>
<div class="grid">
  <div class="card"><h2><span>Live2D 桌宠</span><small>可直接联动动作与语音状态</small></h2><iframe class="live2d-frame" id="live2dFrame" src="/proxy/live2d"></iframe></div>
  <div class="card"><h2><span>智能控制台</span><small>切功能、发动作、让桌宠说话</small></h2><div class="pane">
    <div class="section">
      <div class="row">
        <button onclick="api('/api/start')">启动 / 恢复全部</button>
        <button onclick="api('/api/action/stop')" class="warn">⏹ 停止全部</button>
      </div>
      <div class="badges">
        <div class="badge" id="badge-active">当前功能：待机</div>
        <div class="badge" id="badge-auto">自动切换：未知</div>
        <div class="badge" id="badge-motion">当前动作：-</div>
        <div class="badge" id="badge-resume">恢复倒计时：-</div>
        <div class="badge" id="badge-tts">TTS：空闲</div>
      </div>
    </div>

    <div class="section">
      <div class="section-title" style="margin-bottom:8px">📡 直接跳转（新标签页）</div>
      <div class="row">
        <a href="/proxy/direct/5000/" target="_blank" onclick="launch(5000)"><button style="margin:0">🤚 手势识别 <small style="opacity:.6;font-weight:400">5000</small></button></a>
        <a href="/proxy/direct/8088/" target="_blank" onclick="launch(8088)"><button style="margin:0">🔗 总跟踪 <small style="opacity:.6;font-weight:400">8088</small></button></a>
      </div>
      <div class="row">
        <a href="/proxy/direct/5001/" target="_blank" onclick="launch(5001)"><button style="margin:0">🖱 鼠标控制 <small style="opacity:.6;font-weight:400">5001</small></button></a>
        <a href="/proxy/speaker/" target="_blank" onclick="launch(8095)"><button style="margin:0">🎤 声纹识别 <small style="opacity:.6;font-weight:400">8095</small></button></a>
      </div>
      <div class="row">
        <a href="/proxy/direct/8080/" target="_blank" onclick="launch(8080)"><button style="margin:0">📷 摄像头RKNN <small style="opacity:.6;font-weight:400">8080</small></button></a>
        <a href="/proxy/direct/8083/" target="_blank" onclick="launch(8083)"><button style="margin:0">🎮 绫地宁宁 <small style="opacity:.6;font-weight:400">8083</small></button></a>
      </div>
      <div class="row">
        <a href="/proxy/emotion/" target="_blank" onclick="launchMode('emotion')"><button style="margin:0">😊 情绪识别 <small style="opacity:.6;font-weight:400">8088</small></button></a>
        <button class="secondary" style="margin:0" onclick="api('/api/action/emotion')">启动情绪识别</button>
      </div>
      <div class="small tip" style="margin-top:8px">点击后自动在新标签页打开对应服务，同时后台启动。如果服务没起来，手动刷新一下页面即可。</div>
    </div>

    <div class="section">
      <div class="section-title">桌宠动作</div>
      <div class="viewer-tabs">
        <button class="secondary" onclick="window.open('/proxy/live2d','_blank')">桌宠新标签</button>
        <button class="secondary" onclick="window.open('/proxy/live2d/debug-actions','_blank')">动作调试页</button>
        <button class="secondary" onclick="petAction('wave')">挥手</button>
        <button class="secondary" onclick="petAction('heart')">比心</button>
      </div>
      <div class="row">
        <button class="good" onclick="petAction('auto-on')">开启自动切换</button>
        <button class="warn" onclick="petAction('auto-off')">暂停自动切换</button>
      </div>
    </div>

    <div class="section">
      <div class="section-title">页面语音</div>
      <input id="ttsText" placeholder="输入要说的话，比如：欢迎回来，今天也要加油哦">
      <div class="row">
        <button class="good" onclick="speak()">让桌宠说话</button>
        <button class="secondary" onclick="refresh()">刷新状态</button>
      </div>
      <div class="small tip">说话时会自动暂停动作切换；播放结束后会延迟几秒恢复。</div>

      <div style="height:12px"></div>
      <div class="section-title">USB 麦克风语音输入</div>
      <div class="row">
        <button id="recordTextStartBtn" onclick="startUsbVoiceInput()">🎙 开始语音输入</button>
        <button class="warn" id="recordTextStopBtn" onclick="stopUsbVoiceInput()">⏹ 结束录音</button>
      </div>
      <div class="row">
        <button class="good" onclick="stopUsbVoiceCommand()">执行语音命令</button>
        <button class="warn" id="voiceConfirmBtn" onclick="confirmVoiceCommand()" disabled>确认执行</button>
      </div>
      <div class="small tip">只使用板子上的 USB 麦克风。可说“打开百度”，也可说“写代码...”。写代码和运行命令会先等待确认。</div>
      <pre id="recordTextResult">等待 USB 麦克风语音输入...</pre>
    </div>

    <div class="section">
      <div class="section-title">麦克风输入（TTS / 声纹）</div>
      <div class="row">
        <select id="micDeviceSelect"><option value="">加载设备中...</option></select>
        <input id="micSeconds" type="number" min="1" max="20" value="4" placeholder="录音秒数">
      </div>
      <div class="row">
        <input id="micSpeakerName" placeholder="参考音频人物名，例如：ningning">
        <button class="secondary" onclick="loadMicDevices()">刷新麦克风列表</button>
      </div>
      <div class="row">
        <button onclick="recordMicRef()">🎙 录参考音频</button>
        <button class="good" onclick="recordMicIdentify()">🎧 录音并识别</button>
      </div>
      <div class="small tip">这里直接调用你现在 8095 那套板子本机 arecord 能力，不用再跳去下面找。浏览器不支持 getUserMedia 时，也能继续用这组板载麦克风按钮。</div>
      <pre id="micResult">等待麦克风操作...</pre>
    </div>

    <div class="section">
      <div class="section-title">文本指令</div>
      <input id="text" placeholder="输入指令，比如：切换人脸识别 / 朗读 你好呀">
      <div class="row">
        <button onclick="sendText()">发送指令</button>
        <button class="secondary" onclick="sendVoiceCommandText()">按语音命令执行</button>
      </div>
      <div style="height:10px"></div>
      <textarea id="reply" rows="7" readonly></textarea>
      <div class="panel-note">这里保留最近一次操作返回，方便确认切换、TTS 和联动状态。调试 JSON 默认折叠，避免把页面塞得太乱。</div>
      <button class="secondary" onclick="toggleDebug()">展开 / 隐藏调试信息</button>
      <pre id="status" style="display:none"></pre>
    </div>
  </div></div>
  <div class="card"><h2><span>📡 端口直达</span><small>各服务独立端口，点击即跳转</small></h2><div class="pane">
    <div class="section">
      <div class="section-title">🎮 智能交互</div>
      <div class="row">
        <a href="/proxy/live2d" target="_blank" onclick="launch(5002)"><button style="margin:0">🎭 Live2D 桌宠 <small style="opacity:.6;font-weight:400">5002</small></button></a>
        <a href="/proxy/speaker/" target="_blank" onclick="launch(8095)"><button style="margin:0">🎤 声纹建库 <small style="opacity:.6;font-weight:400">8095</small></button></a>
      </div>
      <div class="row">
        <a href="/proxy/direct/8083/" target="_blank" onclick="launch(8083)"><button style="margin:0">🎮 绫地宁宁 <small style="opacity:.6;font-weight:400">8083</small></button></a>
        <a href="/proxy/direct/9880/" target="_blank" onclick="launch(9880)"><button style="margin:0">🗣 GPT-SoVITS <small style="opacity:.6;font-weight:400">9880</small></button></a>
      </div>
    </div>
    <div class="section">
      <div class="section-title">📷 视觉检测</div>
      <div class="row">
        <a href="/proxy/direct/5000/" target="_blank" onclick="launch(5000)"><button style="margin:0">🤚 手势识别 <small style="opacity:.6;font-weight:400">5000</small></button></a>
        <a href="/proxy/direct/8088/" target="_blank" onclick="launch(8088)"><button style="margin:0">🔗 总跟踪 <small style="opacity:.6;font-weight:400">8088</small></button></a>
      </div>
      <div class="row">
        <a href="/proxy/direct/5001/" target="_blank" onclick="launch(5001)"><button style="margin:0">🖱 鼠标控制 <small style="opacity:.6;font-weight:400">5001</small></button></a>
        <a href="/proxy/direct/8080/" target="_blank" onclick="launch(8080)"><button style="margin:0">📷 摄像头RKNN <small style="opacity:.6;font-weight:400">8080</small></button></a>
      </div>
      <div class="row">
        <a href="/proxy/emotion/" target="_blank" onclick="launchMode('emotion')"><button style="margin:0">😊 情绪识别 <small style="opacity:.6;font-weight:400">8088</small></button></a>
        <button class="secondary" style="margin:0" onclick="api('/api/action/emotion')">启动情绪识别</button>
      </div>
    </div>
  </div></div>
  <div class="card speaker-card"><h2><span>声纹建库与识别</span><small>已整合进 19000，不用单独记 8095</small></h2><iframe id="speakerFrame" src="__SPEAKER_URL__"></iframe></div>
  <div class="card"><h2><span>🧠 AI 问答引擎</span><small>本地NPU / 云端API 双模式</small></h2><div class="pane">
    <div class="section">
      <div class="row">
        <button id="aiModeBtn" class="good" onclick="toggleAIMode()">🖥️ 本地模式</button>
        <button class="secondary" onclick="aiResetStats()">🔄 重置统计</button>
      </div>
      <div class="badges">
        <div class="badge" id="ai-badge-mode">模式：本地</div>
        <div class="badge" id="ai-badge-status">状态：就绪</div>
        <div class="badge" id="ai-badge-speed">速度：-</div>
        <div class="badge" id="ai-badge-stats">请求：0</div>
      </div>
    </div>
    <div class="section">
      <div class="section-title">预设问题</div>
      <div id="aiPresets" style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px"></div>
      <form id="aiAskForm" method="POST" action="/ai/form_ask">
        <input id="aiQuestion" name="q" placeholder="输入你的问题..." style="margin-bottom:8px">
        <button id="aiSendBtn" type="submit" style="margin-top:0">💬 发送</button>
      </form>
      <div style="height:8px"></div>
      <pre id="aiAnswer" style="min-height:80px;max-height:240px;overflow-y:auto;background:#0a1220;padding:12px;border-radius:14px;border:1px solid rgba(255,255,255,.06);font-size:13px;line-height:1.5">等待输入问题...</pre>
    </div>
    <div class="section" id="aiCloudConfig" style="display:none">
      <div class="section-title">🌐 云端设置</div>
      <input id="aiApiUrl" placeholder="API地址，例如 https://api.openai.com/v1/chat/completions" style="margin-bottom:6px">
      <input id="aiApiKey" placeholder="API Key" type="password" style="margin-bottom:6px">
      <input id="aiModelName" placeholder="模型名，例如 gpt-4o-mini" style="margin-bottom:6px">
      <button class="secondary" onclick="aiSaveCloudConfig()">💾 保存云端配置</button>
      <div class="small tip" style="margin-top:8px">切换云端模式后会自动展开配置面板。支持任何 OpenAI 兼容 API。</div>
    </div>
    <div class="section">
      <div class="section-title" style="cursor:pointer" onclick="toggleAIDebug()">📊 AI 性能指标 <span id="aiDebugToggle">▶</span></div>
      <div id="aiDebugPanel" style="display:none">
        <pre id="aiDebugContent" style="min-height:60px;max-height:200px;overflow-y:auto;font-size:12px">加载中...</pre>
      </div>
    </div>
  </div></div>
</div>
<footer>控制页地址：19000。Live2D / 视觉 / TTS / 声纹都从这里统一进入。</footer>
<script>
const live2dFrame = () => document.getElementById('live2dFrame');
const speakerFrame = () => document.getElementById('speakerFrame');
function petAction(action){
  const map = {
    'wave': 'triggerWaveMode',
    'heart': 'triggerHeartMode',
    'auto-on': 'startAutoSwitch',
    'auto-off': 'stopAutoSwitch',
    'idle': 'playIdle'
  };
  const fn = map[action];
  const frame = live2dFrame();
  try {
    if (frame && frame.contentWindow && fn && typeof frame.contentWindow[fn] === 'function') {
      if (action === 'auto-off') frame.contentWindow[fn]('已从 19000 控制台暂停自动切换');
      else frame.contentWindow[fn]();
    }
  } catch (e) {
    console.warn('petAction failed', action, e);
  }
}
async function api(url, data){
  const r = await fetch(url, data ? {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(data)} : {method:'POST'});
  const j = await r.json(); document.getElementById('reply').value = JSON.stringify(j, null, 2); await refresh(); return j;
}
function toggleDebug(){
  const el = document.getElementById('status');
  el.style.display = (el.style.display === 'none' || !el.style.display) ? 'block' : 'none';
}
async function startUsbVoiceInput(){
  document.getElementById('recordTextResult').textContent = '正在输入文本...';
  document.getElementById('recordTextStartBtn').disabled = true;
  try {
    const r = await fetch('/api/record_text_start', { method:'POST' });
    const j = await r.json();
    document.getElementById('recordTextResult').textContent = '正在输入文本...\n' + JSON.stringify(j, null, 2);
  } catch (e) {
    document.getElementById('recordTextResult').textContent = '开始 USB 录音失败: ' + e.message;
    document.getElementById('recordTextStartBtn').disabled = false;
  }
}
async function stopUsbVoiceInput(){
  document.getElementById('recordTextResult').textContent = '正在结束录音并识别文本...';
  try {
    const r = await fetch('/api/record_text_stop', { method:'POST' });
    const j = await r.json();
    if (j && j.text) {
      const t1 = document.getElementById('ttsText');
      const t2 = document.getElementById('text');
      if (t1) t1.value = j.text;
      if (t2) t2.value = j.text;
    }
    document.getElementById('recordTextResult').textContent = JSON.stringify(j, null, 2);
  } catch (e) {
    document.getElementById('recordTextResult').textContent = '结束 USB 录音失败: ' + e.message;
  } finally {
    document.getElementById('recordTextStartBtn').disabled = false;
  }
}
async function stopUsbVoiceCommand(){
  document.getElementById('recordTextResult').textContent = '正在结束录音、识别并执行语音命令...';
  try {
    const r = await fetch('/api/record_voice_command_stop', { method:'POST' });
    const j = await r.json();
    if (j && j.text) {
      const t1 = document.getElementById('ttsText');
      const t2 = document.getElementById('text');
      if (t1) t1.value = j.text;
      if (t2) t2.value = j.text;
    }
    const confirmBtn = document.getElementById('voiceConfirmBtn');
    if (confirmBtn) confirmBtn.disabled = !(j && j.pending);
    document.getElementById('recordTextResult').textContent = JSON.stringify(j, null, 2);
  } catch (e) {
    document.getElementById('recordTextResult').textContent = '语音命令失败: ' + e.message;
  } finally {
    document.getElementById('recordTextStartBtn').disabled = false;
  }
}
async function sendVoiceCommandText(){
  const text = (document.getElementById('text').value || '').trim();
  if (!text) return;
  document.getElementById('recordTextResult').textContent = '正在执行文本命令...';
  const r = await fetch('/api/voice_command', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({text})});
  const j = await r.json();
  const confirmBtn = document.getElementById('voiceConfirmBtn');
  if (confirmBtn) confirmBtn.disabled = !(j && j.pending);
  document.getElementById('recordTextResult').textContent = JSON.stringify(j, null, 2);
}
async function confirmVoiceCommand(){
  document.getElementById('recordTextResult').textContent = '正在确认执行...';
  const r = await fetch('/api/voice_command_confirm', { method:'POST' });
  const j = await r.json();
  const confirmBtn = document.getElementById('voiceConfirmBtn');
  if (confirmBtn) confirmBtn.disabled = true;
  document.getElementById('recordTextResult').textContent = JSON.stringify(j, null, 2);
}
async function speak(){
  const text = document.getElementById('ttsText').value.trim();
  if (!text) return;
  await api('/api/tts', {text});
}
async function sendText(){ await api('/api/text', {text: document.getElementById('text').value}); }
async function loadMicDevices(){
  try {
    const r = await fetch('/api/speaker_status');
    const j = await r.json();
    const sel = document.getElementById('micDeviceSelect');
    const devices = (j && j.capture_devices) || [];
    const def = (j && j.default_capture_device) || '';
    sel.innerHTML = '';
    if (!devices.length) {
      const opt = document.createElement('option');
      opt.value = def || '';
      opt.textContent = def || '未发现设备';
      sel.appendChild(opt);
    } else {
      for (const dev of devices) {
        const opt = document.createElement('option');
        opt.value = dev;
        opt.textContent = dev;
        if (dev === def) opt.selected = true;
        sel.appendChild(opt);
      }
    }
  } catch (e) {
    document.getElementById('micResult').textContent = '读取麦克风设备失败: ' + e.message;
  }
}
async function recordMicRef(){
  const speaker = (document.getElementById('micSpeakerName').value || '').trim();
  if (!speaker) { document.getElementById('micResult').textContent = '请先填写参考音频人物名'; return; }
  const body = {
    speaker,
    device: document.getElementById('micDeviceSelect').value,
    seconds: parseInt(document.getElementById('micSeconds').value || '4', 10),
  };
  document.getElementById('micResult').textContent = '录制参考音频中...';
  const r = await fetch('/api/mic_ref', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  const j = await r.json();
  document.getElementById('micResult').textContent = JSON.stringify(j, null, 2);
}
async function recordMicIdentify(){
  const body = {
    device: document.getElementById('micDeviceSelect').value,
    seconds: parseInt(document.getElementById('micSeconds').value || '4', 10),
    threshold: 0.70,
    margin: 0.00,
    topk: 5,
    segments: 3,
  };
  document.getElementById('micResult').textContent = '录音并识别中...';
  const r = await fetch('/api/mic_identify', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
  const j = await r.json();
  document.getElementById('micResult').textContent = JSON.stringify(j, null, 2);
}
function setBadge(id, text, cls){ const el = document.getElementById(id); el.textContent = text; el.className = 'badge' + (cls ? ' ' + cls : ''); }
function syncPet(status){
  window.__lastStatus = status;
  const pet = status.pet || {};
  const tts = status.tts || {};
  setBadge('badge-active', '当前功能：' + (status.active_title || '待机'));
  setBadge('badge-auto', '自动切换：' + ({on:'开启', paused:'暂停', unknown:'未知'}[pet.auto_mode] || pet.auto_mode || '未知'), pet.auto_mode === 'on' ? 'on' : (pet.auto_mode === 'paused' ? 'off' : ''));
  setBadge('badge-motion', '当前动作：' + (pet.last_motion || '-'));
  setBadge('badge-resume', '恢复倒计时：' + (pet.resume_in_sec != null ? (pet.resume_in_sec + ' 秒') : '-'), pet.resume_in_sec != null ? 'wait' : '');
  setBadge('badge-tts', 'TTS：' + ({idle:'空闲', running:'生成/播放中', done:'完成', error:'失败'}[tts.status] || tts.status || '空闲'), tts.status === 'running' ? 'wait' : (tts.status === 'done' ? 'on' : (tts.status === 'error' ? 'off' : '')));
}
async function refresh(){ const r = await fetch('/api/status'); const j = await r.json(); document.getElementById('status').textContent = JSON.stringify(j, null, 2); syncPet(j); }
window.addEventListener('message', (event) => {
  const data = event && event.data;
  if (!data || data.source !== 'live2d-proxy' || data.type !== 'pet-status') return;
  const payload = data.payload || {};
  const stateView = { pet: { auto_mode: payload.auto || 'unknown', last_motion: payload.lastMotion || '', resume_in_sec: payload.resumeInSec ?? null, speaking: !!payload.speaking } };
  syncPet({ ...(window.__lastStatus || {}), pet: stateView.pet, tts: { status: payload.speaking ? 'running' : ((window.__lastStatus||{}).tts||{}).status || 'idle' } });
});
// ====== AI 引擎 ======
let aiDebugVisible = false;
let aiCurrentMode = 'local';

async function aiAskInline(){
  const input = document.getElementById('aiQuestion');
  const ans = document.getElementById('aiAnswer');
  const btn = document.getElementById('aiSendBtn');
  const speed = document.getElementById('ai-badge-speed');
  const q = input ? input.value.trim() : '';
  if (!q) {
    if (ans) ans.textContent = '请输入问题';
    return;
  }
  const original = btn ? btn.textContent : '';
  const started = Date.now();
  if (btn) { btn.disabled = true; btn.textContent = '本地推理中...'; }
  if (ans) ans.textContent = '已点击发送。正在请求本地 NPU，请等待 30-60 秒...';
  if (speed) speed.textContent = '速度：请求已发送';
  const timer = setInterval(() => {
    const sec = Math.floor((Date.now() - started) / 1000);
    if (ans) ans.textContent = '本地 NPU 正在推理。已用时：' + sec + ' 秒';
    if (speed) speed.textContent = '速度：推理中 ' + sec + 's';
  }, 1000);
  try {
    const r = await fetch('/api/ai/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({q})
    });
    const raw = await r.text();
    let j;
    try { j = JSON.parse(raw); } catch(e) { j = {ok:false, error: raw || ('HTTP ' + r.status)}; }
    if (j.ok) {
      if (ans) ans.textContent = j.answer || '(空回复)';
      if (speed) speed.textContent = '速度：' + j.elapsed + 's' + (j.perf && j.perf.generate_speed_tps ? ' (' + j.perf.generate_speed_tps + ' tok/s)' : '');
    } else {
      if (ans) ans.textContent = '请求失败：' + (j.answer || j.error || raw || r.status);
    }
    try { loadAIStatus(); } catch(e) {}
  } catch(e) {
    if (ans) ans.textContent = '请求失败：' + e.message;
  } finally {
    clearInterval(timer);
    if (btn) { btn.disabled = false; btn.textContent = original || '💬 发送'; }
  }
}

async function loadAIStatus(){
  try {
    const r = await fetch('/api/ai/status');
    const j = await r.json();
    aiCurrentMode = j.mode || 'local';
    const stats = j.stats || {};
    const s = j.status || {};
    const cfg = j.config_safe || {};
    const presets = j.preset_questions || [];
    document.getElementById('ai-badge-mode').textContent = '模式：' + (aiCurrentMode === 'local' ? '🖥️ 本地NPU' : '🌐 云端API');
    document.getElementById('ai-badge-mode').className = 'badge ' + (aiCurrentMode === 'local' ? 'on' : 'wait');
    document.getElementById('ai-badge-status').textContent = (s.local_available ? '🖥️本地就绪' : '') + (s.cloud_available ? ' 🌐云端就绪' : '') || '状态：就绪';
    document.getElementById('ai-badge-stats').textContent = '请求：' + stats.total_requests + ' (本地' + stats.total_local + '/云端' + stats.total_cloud + ')';
    document.getElementById('aiModeBtn').textContent = aiCurrentMode === 'local' ? '🖥️ 本地模式' : '🌐 云端模式';
    document.getElementById('aiModeBtn').className = aiCurrentMode === 'local' ? 'good' : 'warn';
    document.getElementById('aiCloudConfig').style.display = aiCurrentMode === 'cloud' ? 'block' : 'none';
    if (cfg && cfg.cloud) {
      if (cfg.cloud.api_url) document.getElementById('aiApiUrl').value = cfg.cloud.api_url;
      if (cfg.cloud.api_key) document.getElementById('aiApiKey').value = cfg.cloud.api_key;
      if (cfg.cloud.model) document.getElementById('aiModelName').value = cfg.cloud.model;
    }
    // 预设问题
    const container = document.getElementById('aiPresets');
    container.innerHTML = '';
    for (const q of presets) {
      const btn = document.createElement('button');
      btn.className = 'secondary';
      btn.style.cssText = 'margin:0;padding:6px 12px;font-size:12px;width:auto';
      btn.textContent = q;
      btn.onclick = function(){
        document.getElementById('aiQuestion').value = q;
        aiAskInline();
      };
      container.appendChild(btn);
    }
  } catch(e) {
    console.warn('loadAIStatus error', e);
  }
}

async function aiAsk(){
  const q = document.getElementById('aiQuestion').value.trim();
  if (!q) return;
  const ans = document.getElementById('aiAnswer');
  const sendBtn = document.getElementById('aiSendBtn');
  const started = Date.now();
  if (sendBtn) {
    sendBtn.disabled = true;
    sendBtn.dataset.originalText = sendBtn.dataset.originalText || sendBtn.textContent;
    sendBtn.textContent = '本地推理中...';
  }
  ans.textContent = '本地 NPU 正在加载模型并生成，首次响应通常需要 30-60 秒...';
  document.getElementById('ai-badge-speed').textContent = '速度：推理中...';
  const timer = setInterval(() => {
    const sec = Math.floor((Date.now() - started) / 1000);
    ans.textContent = '本地 NPU 正在加载模型并生成，请稍等。已用时：' + sec + ' 秒';
    document.getElementById('ai-badge-speed').textContent = '速度：推理中 ' + sec + 's';
  }, 1000);
  try {
    const r = await fetch('/api/ai/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({q: q})
    });
    const raw = await r.text();
    let j = {};
    try { j = JSON.parse(raw); } catch(e) { j = {ok:false, error: raw || ('HTTP ' + r.status)}; }
    if (j.ok) {
      ans.textContent = j.answer || '(空回复)';
      const modeIcon = j.mode === 'local' ? '🖥️' : '🌐';
      document.getElementById('ai-badge-speed').textContent = '速度：' + j.elapsed + 's' + (j.perf && j.perf.generate_speed_tps ? ' (' + j.perf.generate_speed_tps + ' tok/s)' : '');
    } else {
      ans.textContent = '❌ ' + (j.answer || j.error || '请求失败');
    }
    loadAIStatus();
  } catch(e) {
    ans.textContent = '❌ 请求失败: ' + e.message;
  } finally {
    clearInterval(timer);
    if (sendBtn) {
      sendBtn.disabled = false;
      sendBtn.textContent = sendBtn.dataset.originalText || '💬 发送';
    }
  }
}

async function toggleAIMode(){
  const newMode = aiCurrentMode === 'local' ? 'cloud' : 'local';
  try {
    await fetch('/api/ai/mode', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({mode: newMode})
    });
    aiCurrentMode = newMode;
    loadAIStatus();
  } catch(e) {
    console.warn('toggleAIMode error', e);
  }
}

async function aiSaveCloudConfig(){
  try {
    await fetch('/api/ai/config', {
      method: 'PUT',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        api_url: document.getElementById('aiApiUrl').value.trim(),
        api_key: document.getElementById('aiApiKey').value.trim(),
        model: document.getElementById('aiModelName').value.trim(),
      })
    });
    alert('✅ 云端配置已保存');
    loadAIStatus();
  } catch(e) {
    alert('❌ 保存失败: ' + e.message);
  }
}

async function aiResetStats(){
  await fetch('/api/ai/reset_stats', {method: 'POST'});
  loadAIStatus();
}

function toggleAIDebug(){
  aiDebugVisible = !aiDebugVisible;
  document.getElementById('aiDebugPanel').style.display = aiDebugVisible ? 'block' : 'none';
  document.getElementById('aiDebugToggle').textContent = aiDebugVisible ? '▼' : '▶';
  if (aiDebugVisible) updateAIDebug();
}

async function updateAIDebug(){
  try {
    const r = await fetch('/api/ai/status');
    const j = await r.json();
    document.getElementById('aiDebugContent').textContent = JSON.stringify(j, null, 2);
  } catch(e) {
    document.getElementById('aiDebugContent').textContent = '获取状态失败: ' + e.message;
  }
}

// Keep the normal form action as a no-JS fallback, but use inline fetch when JS is alive.
setTimeout(() => {
  const inp = document.getElementById('aiQuestion');
  if (inp) {
    inp.addEventListener('keydown', function(e){
      if (e.key === 'Enter') {
        e.preventDefault();
        aiAskInline();
      }
    });
  }
  const form = document.getElementById('aiAskForm');
  if (form) {
    form.addEventListener('submit', function(e){
      e.preventDefault();
      aiAskInline();
    });
  }
}, 500);

// ====== 一键启动（后台调用，不干扰跳转） ======
function launch(port){
  // 不等待，直接在后台启动服务
  fetch('/api/launch/' + port, { method: 'POST' }).catch(() => {});
}
function launchMode(name){
  fetch('/api/action/' + name, { method: 'POST' }).catch(() => {});
}

loadAIStatus();
setInterval(refresh, 3000); refresh(); loadMicDevices();
</script></body></html>'''.replace('__LIVE2D_URL__', LIVE2D_URL).replace('__SPEAKER_URL__', SPEAKER_URL)


@app.route('/')
def index():
    return page()


@app.route('/api/status')
def status():
    global TTS_PROC
    if TTS_PROC is not None:
        code = TTS_PROC.poll()
        if code is None:
            state['tts']['status'] = 'running'
            state['pet']['speaking'] = True
        else:
            if state['tts']['status'] == 'running':
                state['tts']['status'] = 'done' if code == 0 else 'error'
                state['tts']['ended_at'] = time.time()
                state['tts']['code'] = code
            state['pet']['speaking'] = False
            TTS_PROC = None
    ensure_all()
    return jsonify(state)


def direct_upstream_html(port, path='/'):
    r = requests.get(f'http://127.0.0.1:{port}{path}', timeout=15)
    html = r.text
    prefix = f'/proxy/direct/{port}'
    replacements = [
        ('src="/','src="' + prefix + '/'),
        ('src="/video_feed?', 'src="' + prefix + '/video_feed?'),
        ("src='/", "src='" + prefix + '/'),
        ('href="/','href="' + prefix + '/'),
        ("href='/", "href='" + prefix + '/'),
        ('fetch("/','fetch("' + prefix + '/'),
        ("fetch('/", "fetch('" + prefix + '/'),
        ('action="/','action="' + prefix + '/'),
        ("action='/", "action='" + prefix + '/'),
    ]
    for old, new in replacements:
        html = html.replace(old, new)
    headers = proxy_headers(r.headers)
    headers['Content-Type'] = headers.get('Content-Type', 'text/html; charset=utf-8')
    return html, r.status_code, headers


def loading_page(port, name=None):
    title = name or PORT_NAMES.get(port, f'端口 {port}')
    html = f'''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><meta http-equiv="refresh" content="3"><title>正在启动 {title}</title><style>body{{background:#0b1020;color:#e9eefc;display:flex;justify-content:center;align-items:center;height:100vh;font-family:system-ui;flex-direction:column;gap:16px;text-align:center;padding:20px}}h1{{font-size:24px;font-weight:400}}.spinner{{width:40px;height:40px;border:4px solid rgba(255,255,255,.1);border-top-color:#5b8cff;border-radius:50%;animation:spin .8s linear infinite}}@keyframes spin{{to{{transform:rotate(360deg)}}}}</style></head><body><div class="spinner"></div><h1>正在启动 {title}…</h1><p style="color:#9fb0d4;font-size:14px">端口 {port} · 自动刷新，请稍候</p></body></html>'''
    return html, 200, {'Content-Type': 'text/html; charset=utf-8'}


def ensure_named_mode(mode, port):
    if state.get('active_mode') == mode and wait_port(port, timeout=1.0):
        return True
    result = start_mode_process(mode, wait_ready=True)
    return bool(result.get('ready'))


def _ensure_port(port):
    """确保指定端口的服务正在运行。启动新服务前，先关掉所有其他占摄像头的视觉进程。"""
    # 如果目标端口已经活着，直接返回
    try:
        r = requests.get(f'http://127.0.0.1:{port}/', timeout=2)
        if r.status_code < 500:
            return True
    except:
        pass

    direct_modes = {
        5000: 'hand',
        5001: 'mouse_aux',
        8088: 'track',
    }
    if port in direct_modes:
        return ensure_named_mode(direct_modes[port], port)

    # 如果是视觉类端口（占摄像头的），先杀掉所有其他视觉进程
    VISUAL_PORTS = {5000, 5001, 8080, 8083, 8088}
    if port in VISUAL_PORTS:
        stop_live2d_gesture_camera()
        for old_port in VISUAL_PORTS:
            if old_port == port:
                continue
            try:
                requests.get(f'http://127.0.0.1:{old_port}/', timeout=1)
                # 这个端口还在跑，杀掉对应的进程
                subprocess.run(f"fuser -k {old_port}/tcp 2>/dev/null", shell=True, timeout=3)
            except:
                pass
        # 也杀掉任何残留的python视觉进程
        subprocess.run(f"pkill -9 -f '{VISION_PATTERNS}' 2>/dev/null", shell=True, timeout=3)
        subprocess.run('pkill -9 -f "rknn_camera_simulator|ningning_simple|zongtitrack4|emotion_camera|media4|mediapipe_mouse" 2>/dev/null', shell=True, timeout=3)
        time.sleep(1)
    else:
        # 非视觉类（5002/8095/9880），只关了对应的旧进程
        try:
            subprocess.run(f"fuser -k {port}/tcp 2>/dev/null", shell=True, timeout=3)
        except:
            pass

    # 启动目标服务
    PORT_CMDS = {
        5000: 'python3 /userdata/media4.py',
        5001: 'python3 /userdata/mediapipe_mouse_control.py',
        5002: None,  # 由 ensure_all 处理
        8080: 'bash /home/linaro/.openclaw/workspace/start_rknn_system.sh',
        8083: 'bash /home/linaro/.openclaw/workspace/start_ningning_complete.sh',
        8088: 'python3 /userdata/zongtitrack4.py',
        8095: None,  # 由 ensure_all 处理
        9880: None,  # 由 ensure_gpt_sovits 处理
    }
    cmd = PORT_CMDS.get(port)
    if cmd:
        wd = '/userdata' if port in (5000, 5001, 8088) else '/home/linaro/.openclaw/workspace'
        try:
            subprocess.Popen(cmd, shell=True, cwd=wd, stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           preexec_fn=os.setsid)
        except Exception as e:
            print(f'_ensure_port start failed: {e}')
    # 对于5002/8095/9880，调用 ensure_all
    if port in (5002, 8095, 9880):
        try:
            ensure_all()
        except:
            pass
    # 等待端口就绪 (最多20秒)
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            r = requests.get(f'http://127.0.0.1:{port}/', timeout=2)
            if r.status_code < 500:
                return True
        except:
            pass
        time.sleep(0.5)
    return False


@app.route('/proxy/direct/<int:port>/', defaults={'subpath': ''}, methods=['GET', 'POST'])
@app.route('/proxy/direct/<int:port>/<path:subpath>', methods=['GET', 'POST'])
def proxy_direct(port, subpath):
    ALLOWED_PROXY_PORTS = {5000, 5001, 5002, 8080, 8083, 8088, 8095, 9880}
    if port not in ALLOWED_PROXY_PORTS:
        return jsonify({'ok': False, 'error': 'unsupported_port'}), 400
    # 确保服务运行
    ready = _ensure_port(port)
    target = f'http://127.0.0.1:{port}/' + subpath
    if not ready:
        name = PORT_NAMES.get(port, f'端口 {port}')
        return loading_page(port, name)
    # 流式子路径（video_feed / stream）单独处理——不等完整响应，直接流式透传
    IS_STREAM = subpath in {'stream', 'video_feed'} or 'video_feed' in subpath or 'stream' in subpath
    if request.method == 'GET':
        if not subpath:
            return direct_upstream_html(port, '/')
        if IS_STREAM:
            upstream = requests.get(target, params=request.args, stream=True, timeout=60)
            def gen():
                try:
                    for chunk in upstream.iter_content(chunk_size=8192):
                        if chunk:
                            yield chunk
                finally:
                    upstream.close()
            ctype = upstream.headers.get('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            return Response(gen(), status=upstream.status_code, content_type=ctype)
        r = requests.get(target, params=request.args, stream=False, timeout=30)
        ctype = r.headers.get('Content-Type', '')
        if 'text/html' in ctype:
            html = r.text
            prefix = f'/proxy/direct/{port}'
            for old, new in [
                ('src="/','src="' + prefix + '/'),
                ('src="/video_feed?', 'src="' + prefix + '/video_feed?'),
                ("src='/", "src='" + prefix + '/'),
                ('href="/','href="' + prefix + '/'),
                ("href='/", "href='" + prefix + '/'),
                ('fetch("/','fetch("' + prefix + '/'),
                ("fetch('/", "fetch('" + prefix + '/'),
                ('action="/','action="' + prefix + '/'),
                ("action='/", "action='" + prefix + '/'),
            ]:
                html = html.replace(old, new)
            headers = proxy_headers(r.headers)
            headers['Content-Type'] = headers.get('Content-Type', 'text/html; charset=utf-8')
            return html, r.status_code, headers
        if 'multipart/x-mixed-replace' in ctype:
            def gen():
                try:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            yield chunk
                finally:
                    r.close()
            return Response(gen(), status=r.status_code, content_type=ctype or 'multipart/x-mixed-replace; boundary=frame')
        return Response(r.content, status=r.status_code, headers=proxy_headers(r.headers), content_type=ctype or None)
    # POST 请求
    r = requests.request(request.method, target, params=request.args, json=request.get_json(silent=True), data=None if request.is_json else request.get_data(), stream=IS_STREAM, timeout=60)
    ctype = r.headers.get('Content-Type', '')
    if 'multipart/x-mixed-replace' in ctype or IS_STREAM:
        def gen():
            try:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        yield chunk
            finally:
                r.close()
        return Response(gen(), status=r.status_code, content_type=ctype or 'multipart/x-mixed-replace; boundary=frame')
    return Response(r.content, status=r.status_code, headers=proxy_headers(r.headers), content_type=ctype or None)



@app.route('/proxy/live2d')
@functools.lru_cache(maxsize=1)
def proxy_live2d():
    ensure_all()
    if not wait_port(5002, timeout=1.0):
        return loading_page(5002, 'Live2D 桌宠')
    r = requests.get('http://127.0.0.1:5002/', timeout=15)
    html = rewrite_live2d_html(r.text)
    headers = proxy_headers(r.headers)
    headers['Content-Type'] = headers.get('Content-Type', 'text/html; charset=utf-8')
    return html, r.status_code, headers


@app.route('/proxy/live2d/<path:subpath>', methods=['GET', 'POST', 'PUT'])
def proxy_live2d_any(subpath):
    ensure_all()
    if not wait_port(5002, timeout=1.0):
        return loading_page(5002, 'Live2D 桌宠')
    target = f'http://127.0.0.1:5002/{subpath}'
    if request.method == 'GET':
        r = requests.get(target, params=request.args, timeout=60, stream=True)
    else:
        r = requests.request(request.method, target, params=request.args, data=request.get_data(), timeout=60, stream=True)
    ctype = r.headers.get('Content-Type', '')
    if 'text/html' in ctype:
        html = rewrite_live2d_html(r.text)
        headers = proxy_headers(r.headers)
        headers['Content-Type'] = headers.get('Content-Type', 'text/html; charset=utf-8')
        return html, r.status_code, headers
    if 'multipart/x-mixed-replace' in ctype or '/video_feed' in subpath:
        def generate():
            try:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        yield chunk
            finally:
                r.close()
        return Response(generate(), status=r.status_code, content_type=ctype or 'multipart/x-mixed-replace; boundary=frame')
    return Response(r.content, status=r.status_code, headers=proxy_headers(r.headers))


@app.route('/proxy/live2d/runtime/<path:asset_path>')
def proxy_live2d_runtime(asset_path):
    r = requests.get(f'http://127.0.0.1:5002/runtime/{asset_path}', timeout=15)
    return Response(r.content, status=r.status_code, headers=proxy_headers(r.headers))


@app.route('/proxy/live2d/model/<path:asset_path>')
def proxy_live2d_model(asset_path):
    r = requests.get(f'http://127.0.0.1:5002/model/{asset_path}', timeout=15)
    return Response(r.content, status=r.status_code, headers=proxy_headers(r.headers))


@app.route('/proxy/hand/')
def proxy_hand():
    if not ensure_named_mode('hand', 5000):
        return loading_page(5000, '手势识别')
    r = requests.get('http://127.0.0.1:5000/', timeout=15)
    html = rewrite_generic_html(r.text, [('src="/video_feed"','src="/proxy/hand/video_feed"'), ("src='/video_feed'","src='/proxy/hand/video_feed'")])
    headers = proxy_headers(r.headers)
    headers['Content-Type'] = headers.get('Content-Type', 'text/html; charset=utf-8')
    return html, r.status_code, headers


@app.route('/proxy/hand/video_feed')
def proxy_hand_video_feed():
    if not ensure_named_mode('hand', 5000):
        return loading_page(5000, '手势识别')
    upstream = requests.get('http://127.0.0.1:5000/video_feed', stream=True, timeout=30)
    def generate():
        try:
            for chunk in upstream.iter_content(chunk_size=8192):
                if chunk:
                    yield chunk
        finally:
            upstream.close()
    return Response(generate(), status=upstream.status_code, content_type=upstream.headers.get('Content-Type', 'multipart/x-mixed-replace; boundary=frame'))


def proxy_8088_page():
    r = requests.get('http://127.0.0.1:8088/', timeout=15)
    html = rewrite_generic_html(r.text, [
        ('src="/video_feed"', 'src="/proxy/mode/video_feed"'),
        ("src='/video_feed'", "src='/proxy/mode/video_feed'"),
        ('src="/stream"', 'src="/proxy/mode/stream"'),
        ("src='/stream'", "src='/proxy/mode/stream'"),
        ('fetch("/click"', 'fetch("/proxy/mode/click"'),
        ("fetch('/click'", "fetch('/proxy/mode/click'"),
        ('fetch("/start_capture"', 'fetch("/proxy/mode/start_capture"'),
        ("fetch('/start_capture'", "fetch('/proxy/mode/start_capture'"),
        ('fetch("/capture_photo"', 'fetch("/proxy/mode/capture_photo"'),
        ("fetch('/capture_photo'", "fetch('/proxy/mode/capture_photo'"),
        ('fetch("/delete_user"', 'fetch("/proxy/mode/delete_user"'),
        ("fetch('/delete_user'", "fetch('/proxy/mode/delete_user'"),
        ('fetch("/get_user_list"', 'fetch("/proxy/mode/get_user_list"'),
        ("fetch('/get_user_list'", "fetch('/proxy/mode/get_user_list'"),
        ('fetch("/analyze"', 'fetch("/proxy/mode/analyze"'),
        ("fetch('/analyze'", "fetch('/proxy/mode/analyze'"),
    ])
    headers = proxy_headers(r.headers)
    headers['Content-Type'] = headers.get('Content-Type', 'text/html; charset=utf-8')
    return html, r.status_code, headers


def proxy_emotion_page():
    r = requests.get('http://127.0.0.1:8088/', timeout=15)
    prefix = '/proxy/emotion'
    html = rewrite_generic_html(r.text, [
        ('src="/snapshot.jpg', f'src="{prefix}/snapshot.jpg'),
        ("src='/snapshot.jpg", f"src='{prefix}/snapshot.jpg"),
        ("cam.src = '/snapshot.jpg", f"cam.src = '{prefix}/snapshot.jpg"),
        ('fetch("/api/', f'fetch("{prefix}/api/'),
        ("fetch('/api/", f"fetch('{prefix}/api/"),
    ])
    headers = proxy_headers(r.headers)
    headers['Content-Type'] = headers.get('Content-Type', 'text/html; charset=utf-8')
    return html, r.status_code, headers


@app.route('/proxy/head/')
def proxy_head():
    if not ensure_named_mode('head', 8088):
        return loading_page(8088)
    return proxy_8088_page()


@app.route('/proxy/face/')
def proxy_face():
    if not ensure_named_mode('face', 8088):
        return loading_page(8088)
    return proxy_8088_page()


@app.route('/proxy/track/')
def proxy_track():
    if not ensure_named_mode('track', 8088):
        return loading_page(8088)
    return proxy_8088_page()


@app.route('/proxy/emotion/')
def proxy_emotion():
    if not ensure_named_mode('emotion', 8088):
        return loading_page(8088, '情绪识别')
    return proxy_emotion_page()


@app.route('/proxy/emotion/<path:subpath>', methods=['GET', 'POST'])
def proxy_emotion_any(subpath):
    if not ensure_named_mode('emotion', 8088):
        return loading_page(8088, '情绪识别')
    target = f'http://127.0.0.1:8088/{subpath}'
    if request.method == 'POST':
        r = requests.post(target, params=request.args, json=request.get_json(silent=True), data=None if request.is_json else request.get_data(), timeout=60)
    else:
        r = requests.get(target, params=request.args, timeout=60, stream=('video_feed' in subpath))
    ctype = r.headers.get('Content-Type', '')
    if 'text/html' in ctype:
        return proxy_emotion_page()
    if 'multipart/x-mixed-replace' in ctype or 'video_feed' in subpath:
        def generate():
            try:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        yield chunk
            finally:
                r.close()
        return Response(generate(), status=r.status_code, content_type=ctype or 'multipart/x-mixed-replace; boundary=frame')
    return Response(r.content, status=r.status_code, headers=proxy_headers(r.headers), content_type=ctype or None)


@app.route('/proxy/mode/video_feed')
def proxy_mode_video_feed():
    if not wait_port(8088, timeout=1.0):
        return loading_page(8088, '视觉检测')
    upstream = requests.get('http://127.0.0.1:8088/video_feed', stream=True, timeout=30)
    def generate():
        try:
            for chunk in upstream.iter_content(chunk_size=8192):
                if chunk:
                    yield chunk
        finally:
            upstream.close()
    return Response(generate(), status=upstream.status_code, content_type=upstream.headers.get('Content-Type', 'multipart/x-mixed-replace; boundary=frame'))


@app.route('/proxy/mode/stream')
def proxy_mode_stream():
    if not wait_port(8088, timeout=1.0):
        return loading_page(8088, '视觉检测')
    upstream = requests.get('http://127.0.0.1:8088/video_feed', stream=True, timeout=30)
    def generate():
        try:
            for chunk in upstream.iter_content(chunk_size=8192):
                if chunk:
                    yield chunk
        finally:
            upstream.close()
    return Response(generate(), status=upstream.status_code, content_type=upstream.headers.get('Content-Type', 'multipart/x-mixed-replace; boundary=frame'))


def _proxy_mode_req(path):
    if not wait_port(8088, timeout=1.0):
        return loading_page(8088, '视觉检测')
    if request.method == 'POST':
        r = requests.post(f'http://127.0.0.1:8088/{path}', json=request.get_json(silent=True) or {}, timeout=30)
    else:
        r = requests.get(f'http://127.0.0.1:8088/{path}', params=request.args, timeout=30)
    return Response(r.content, status=r.status_code, headers=proxy_headers(r.headers))


@app.route('/proxy/mode/click', methods=['POST'])
def proxy_mode_click():
    return _proxy_mode_req('click')


@app.route('/proxy/mode/start_capture')
def proxy_mode_start_capture():
    return _proxy_mode_req('start_capture')


@app.route('/proxy/mode/capture_photo')
def proxy_mode_capture_photo():
    return _proxy_mode_req('capture_photo')


@app.route('/proxy/mode/delete_user')
def proxy_mode_delete_user():
    return _proxy_mode_req('delete_user')


@app.route('/proxy/mode/get_user_list')
def proxy_mode_get_user_list():
    return _proxy_mode_req('get_user_list')


@app.route('/proxy/mode/analyze', methods=['POST'])
def proxy_mode_analyze():
    return _proxy_mode_req('analyze')


@app.route('/proxy/mouse_aux/')
def proxy_mouse_aux():
    if not ensure_named_mode('mouse_aux', 5001):
        return loading_page(5001, '鼠标控制')
    r = requests.get('http://127.0.0.1:5001/', timeout=15)
    html = rewrite_mediapipe_mouse_html(r.text)
    headers = proxy_headers(r.headers)
    headers['Content-Type'] = headers.get('Content-Type', 'text/html; charset=utf-8')
    return html, r.status_code, headers


@app.route('/proxy/mouse_aux/video_feed')
def proxy_mouse_aux_video_feed():
    if not ensure_named_mode('mouse_aux', 5001):
        return loading_page(5001, '鼠标控制')
    upstream = requests.get('http://127.0.0.1:5001/video_feed', stream=True, timeout=30)
    def generate():
        try:
            for chunk in upstream.iter_content(chunk_size=8192):
                if chunk:
                    yield chunk
        finally:
            upstream.close()
    return Response(generate(), status=upstream.status_code, content_type=upstream.headers.get('Content-Type', 'multipart/x-mixed-replace; boundary=frame'))


@app.route('/proxy/mouse_aux/config', methods=['GET', 'POST'])
def proxy_mouse_aux_config():
    if not ensure_named_mode('mouse_aux', 5001):
        return loading_page(5001, '鼠标控制')
    if request.method == 'POST':
        r = requests.post('http://127.0.0.1:5001/config', json=request.get_json(silent=True) or {}, timeout=15)
    else:
        r = requests.get('http://127.0.0.1:5001/config', timeout=15)
    return Response(r.content, status=r.status_code, headers=proxy_headers(r.headers))


@app.route('/proxy/mouse/')
def proxy_mouse():
    return proxy_mouse_aux()


@app.route('/proxy/mouse/video_feed')
def proxy_mouse_video_feed():
    return proxy_mouse_aux_video_feed()


@app.route('/proxy/mouse/config', methods=['GET', 'POST'])
def proxy_mouse_config():
    return proxy_mouse_aux_config()


@app.route('/proxy/speaker/', defaults={'subpath': ''}, methods=['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'])
@app.route('/proxy/speaker/<path:subpath>', methods=['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'])
def proxy_speaker(subpath):
    ensure_all()
    return speaker_proxy_request(request.method, subpath)


@app.route('/api/speaker_status')
def speaker_status():
    ensure_all()
    r = requests.get('http://127.0.0.1:8095/api/status', timeout=30)
    return Response(r.content, status=r.status_code, headers=proxy_headers(r.headers))


@app.route('/api/mic_ref', methods=['POST'])
def mic_ref():
    ensure_all()
    r = requests.post('http://127.0.0.1:8095/api/record_ref_host', json=request.get_json(silent=True) or {}, timeout=120)
    return Response(r.content, status=r.status_code, headers=proxy_headers(r.headers))


@app.route('/api/mic_identify', methods=['POST'])
def mic_identify():
    ensure_all()
    r = requests.post('http://127.0.0.1:8095/api/record_query_host_identify', json=request.get_json(silent=True) or {}, timeout=180)
    return Response(r.content, status=r.status_code, headers=proxy_headers(r.headers))


@app.route('/api/emotion_remote', methods=['POST'])
def api_emotion_remote():
    ensure_all()
    data = request.get_data()
    ctype = request.content_type or ''
    headers = {k:v for k,v in request.headers if k.lower() not in {'host','content-length','content-type'}}
    r = requests.post('http://127.0.0.1:8095/api/emotion_remote', data=data, headers={**headers, 'Content-Type': ctype}, timeout=120)
    return Response(r.content, status=r.status_code, headers=proxy_headers(r.headers))


@app.route('/api/record_host_emotion_remote', methods=['POST'])
def api_record_host_emotion_remote():
    ensure_all()
    r = requests.post('http://127.0.0.1:8095/api/record_host_emotion_remote', json=request.get_json(silent=True) or {}, timeout=180)
    return Response(r.content, status=r.status_code, headers=proxy_headers(r.headers))


@app.route('/api/asr_upload', methods=['POST'])
def asr_upload():
    ensure_all()
    f = request.files.get('file')
    if not f:
        return jsonify({'ok': False, 'error': 'no_file'}), 400
    raw_path = wav_path = None
    try:
        suffix = Path(f.filename or 'upload.webm').suffix or '.webm'
        fd, raw_path = tempfile.mkstemp(prefix='dashboard_asr_', suffix=suffix)
        os.close(fd)
        f.save(raw_path)
        wav_path = raw_path.rsplit('.', 1)[0] + '.wav'
        subprocess.check_call(['ffmpeg', '-y', '-i', raw_path, '-ac', '1', '-ar', '16000', '-vn', wav_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        model = load_asr_model()
        text = ''
        if model:
            try:
                res = model.generate(input=wav_path, batch_size_s=300)
                if res and 'text' in res[0]:
                    text = (res[0]['text'] or '').strip().replace(' ', '')
            except Exception:
                text = ''
        if text and (len(text) < MIN_VALID_LENGTH or set(text).issubset(USELESS_WORDS)):
            text = ''
        state['last_text'] = text
        return jsonify({'ok': True, 'text': text, 'message': '识别完成' if text else '未识别到有效文本'})
    except Exception as e:
        return jsonify({'ok': False, 'error': 'asr_failed', 'message': str(e)})
    finally:
        for p in [raw_path, wav_path]:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


@app.route('/api/record_text_start', methods=['POST'])
def record_text_start():
    device = detect_usb_record_device()
    wav_path = '/tmp/dashboard_record_text.wav'
    pid_path = '/tmp/dashboard_record_text.pid'
    old_pid = None
    if os.path.exists(pid_path):
        try:
            old_pid = int(Path(pid_path).read_text(encoding='utf-8').strip())
        except Exception:
            old_pid = None
    if old_pid:
        try:
            os.killpg(os.getpgid(old_pid), signal.SIGTERM)
        except Exception:
            try:
                os.kill(old_pid, signal.SIGTERM)
            except Exception:
                pass
        time.sleep(0.3)
    for p in [wav_path, pid_path]:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass
    proc = subprocess.Popen(['arecord', '-D', device, '-f', 'S16_LE', '-c', '1', '-r', '16000', wav_path], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
    Path(pid_path).write_text(str(proc.pid), encoding='utf-8')
    return jsonify({'ok': True, 'message': '开始录音', 'pid': proc.pid, 'device': device})


@app.route('/api/record_text_stop', methods=['POST'])
def record_text_stop():
    pid_path = '/tmp/dashboard_record_text.pid'
    wav_path = '/tmp/dashboard_record_text.wav'
    pid = None
    if os.path.exists(pid_path):
        try:
            pid = int(Path(pid_path).read_text(encoding='utf-8').strip())
        except Exception:
            pid = None
    if pid:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except Exception:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
    time.sleep(0.8)
    if not os.path.exists(wav_path) or os.path.getsize(wav_path) <= 0:
        return jsonify({'ok': False, 'error': 'record_failed', 'message': '录音文件为空或未生成'})
    model = load_asr_model()
    text = ''
    if model:
        try:
            res = model.generate(input=wav_path, batch_size_s=300)
            if res and 'text' in res[0]:
                text = (res[0]['text'] or '').strip().replace(' ', '')
        except Exception:
            text = ''
    if text and (len(text) < MIN_VALID_LENGTH or set(text).issubset(USELESS_WORDS)):
        text = ''
    state['last_text'] = text
    return jsonify({'ok': True, 'text': text, 'message': '录音结束，识别完成' if text else '录音结束，但没有识别出有效文本'})


@app.route('/api/voice_command', methods=['POST'])
def voice_command():
    global PENDING_VOICE_COMMAND
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get('text') or '').strip()
    if not text:
        return jsonify({'ok': False, 'error': 'empty', 'message': '没有语音文本'})
    state['last_text'] = text
    plan = plan_voice_command(text)
    if voice_plan_needs_confirmation(plan):
        PENDING_VOICE_COMMAND = {'plan': plan, 'text': text, 'created_at': time.time()}
        message = plan.get('reply') or '这个操作需要确认'
        speak_reply_async(message)
        return jsonify({'ok': True, 'pending': True, 'text': text, 'plan': plan, 'message': message})
    ok, message = execute_voice_plan(plan)
    speak_reply_async(message)
    return jsonify({'ok': ok, 'pending': False, 'text': text, 'plan': plan, 'message': message})


@app.route('/api/voice_command_confirm', methods=['POST'])
def voice_command_confirm():
    global PENDING_VOICE_COMMAND
    pending = PENDING_VOICE_COMMAND
    if not pending:
        return jsonify({'ok': False, 'error': 'no_pending', 'message': '没有等待确认的语音命令'})
    PENDING_VOICE_COMMAND = None
    ok, message = execute_voice_plan(pending.get('plan') or {})
    speak_reply_async(message)
    return jsonify({'ok': ok, 'pending': False, 'text': pending.get('text'), 'plan': pending.get('plan'), 'message': message})


@app.route('/api/record_voice_command_stop', methods=['POST'])
def record_voice_command_stop():
    resp = record_text_stop()
    data = resp.get_json(silent=True) if hasattr(resp, 'get_json') else None
    if not data or not data.get('ok') or not data.get('text'):
        return resp
    with app.test_request_context('/api/voice_command', method='POST', json={'text': data.get('text')}):
        return voice_command()


def kill_visual_processes():
    global VISUAL_PROC
    if VISUAL_PROC and VISUAL_PROC.poll() is None:
        try:
            os.killpg(os.getpgid(VISUAL_PROC.pid), signal.SIGKILL)
        except Exception:
            try:
                VISUAL_PROC.kill()
            except Exception:
                pass
        try:
            VISUAL_PROC.wait(timeout=5)
        except Exception:
            pass
    VISUAL_PROC = None
    subprocess.run(f"pkill -9 -f '{VISION_PATTERNS}'", shell=True)


def wait_port_closed(port, timeout=8.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            requests.get(f'http://127.0.0.1:{port}/', timeout=0.8)
            time.sleep(0.2)
            continue
        except Exception:
            return True
    return False


def start_mode_process(name, wait_ready=True):
    global VISUAL_PROC
    mapping = {
        'hand': ('python3 /userdata/media4.py', '手势识别', 'hand', '手势识别中', 5000),
        'head': ('python3 /userdata/camera44.py', '人头检测', 'head', '人头检测中', 8088),
        'face': ('python3 /userdata/renlian15.py', '人脸识别', 'face', '人脸识别中', 8088),
        'track': ('python3 /userdata/zongtitrack4.py', '总跟踪', 'track', '总跟踪中', 8088),
        'emotion': ('python3 /userdata/emotion_camera.py', '情绪识别', 'emotion', '情绪识别中', 8088),
        'voice': ('python3 /userdata/tts3.py', 'AI语音对话', 'voice', 'AI语音对话中', None),
        'mouse_aux': ('python3 /userdata/mediapipe_mouse_control.py', '鼠标控制', 'mouse_aux', '鼠标控制中', 5001),
    }
    if name not in mapping:
        return {'ok': False, 'error': 'unknown action'}
    stop_live2d_gesture_camera()
    kill_visual_processes()
    cmd, title, mode, active_title, port = mapping[name]
    for old_port in (5000, 5001, 8088):
        wait_port_closed(old_port, timeout=3.0)
    if name == 'voice':
        subprocess.Popen(f"x-terminal-emulator -e bash -lc 'cd /userdata && {cmd}; exec bash'", shell=True, preexec_fn=os.setsid)
        ready = True
    else:
        VISUAL_PROC = subprocess.Popen(cmd, shell=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, preexec_fn=os.setsid)
        ready = wait_port(port, timeout=20.0) if (port and wait_ready) else True
    state['last_action'] = name
    state['active_mode'] = mode
    state['active_title'] = active_title
    msg = f'已切换到 {title}' if ready else f'已启动 {title}，但目标端口 {port} 未就绪'
    return {'ok': ready, 'message': msg, 'mode': mode, 'port': port, 'ready': ready, 'wait_ready': wait_ready}


@app.route('/api/start', methods=['POST'])
def start_all():
    deps = ensure_stack_dependencies()
    kill_visual_processes()
    live2d_camera = start_live2d_gesture_camera()
    state['status'] = 'started'
    state['last_action'] = 'start_all'
    state['active_mode'] = 'idle'
    state['active_title'] = 'Live2D 待机中'
    ok = all([deps['live2d'], deps['speaker'], deps['gpt_sovits'], live2d_camera])
    return jsonify({
        'ok': ok,
        'message': '已启动/恢复基础服务，Live2D 手势摄像头已常开',
        'live2d': deps['live2d'],
        'speaker': deps['speaker'],
        'gpt_sovits': deps['gpt_sovits'],
        'live2d_camera': live2d_camera,
        'mouse_aux': False,
        'track': False,
    })


@app.route('/api/action/<name>', methods=['POST'])
def action(name):
    ensure_all()
    if name == 'stop':
        kill_visual_processes()
        live2d_camera = start_live2d_gesture_camera()
        state['last_action'] = 'stop_all'
        state['active_mode'] = 'idle'
        state['active_title'] = 'Live2D 待机中'
        return jsonify({'ok': True, 'message': '已停止其它视觉，Live2D 手势摄像头已恢复', 'live2d_camera': live2d_camera})
    return jsonify(start_mode_process(name))


@app.route('/api/tts', methods=['POST'])
def tts():
    global TTS_PROC
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get('text') or request.environ.get('web_tts_text') or '').strip()
    state['last_text'] = text
    if not text:
        return jsonify({'ok': False, 'error': 'empty'})
    if TTS_PROC and TTS_PROC.poll() is None:
        return jsonify({'ok': False, 'error': 'busy', 'message': '已有一条语音正在生成/播放'})
    try:
        TTS_PROC, meta = play_tts_http(text)
    except Exception as e:
        state['tts'] = {'status': 'error', 'text': text, 'started_at': None, 'ended_at': None, 'code': None, 'error': str(e)}
        state['pet']['speaking'] = False
        return jsonify({'ok': False, 'error': 'spawn_failed', 'message': str(e)})
    state['tts'] = {'status': 'running', 'text': text, 'started_at': time.time(), 'ended_at': None, 'code': None, 'error': '', 'meta': meta}
    state['pet']['speaking'] = True
    state['last_action'] = 'web_tts'
    return jsonify({'ok': True, 'message': '已开始网页 TTS', 'text': text, 'pid': TTS_PROC.pid, 'meta': meta})


@app.route('/api/text', methods=['POST'])
def text():
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get('text') or '').strip()
    state['last_text'] = text
    if not text:
        return jsonify({'ok': False, 'error': 'empty'})

    if text.startswith('朗读 ') or text.startswith('说 ') or text.startswith('念 '):
        speak_text = text.split(' ', 1)[1].strip() if ' ' in text else ''
        if speak_text:
            request.environ['web_tts_text'] = speak_text
            return tts()

    if any(k in text for k in ['手势']):
        return action('hand')
    if any(k in text for k in ['人头']):
        return action('head')
    if any(k in text for k in ['人脸']):
        return action('face')
    if any(k in text for k in ['跟踪', '追踪']):
        return action('track')
    if any(k in text for k in ['鼠标']):
        return action('mouse_aux')
    if any(k in text for k in ['打开百度', '百度一下', '搜索百度']):
        return jsonify({'ok': True, 'message': '已识别到打开百度指令', 'action': 'open_baidu', 'url': 'https://www.baidu.com'})
    if any(k in text for k in ['语音', '说话', '朗读']):
        return jsonify({'ok': True, 'message': '请直接使用页面里的 TTS 输入框和按钮', 'text': text})
    return jsonify({'ok': True, 'message': f'收到文本指令：{text}', 'text': text})


# ============================================================
# AI 引擎 API
# ============================================================

def run_ai_prompt(prompt, mode=None):
    prompt = (prompt or '').strip()
    mode = mode or ai_state.get('mode', 'local')
    if not prompt:
        return {'ok': False, 'error': '请输入问题', 'answer': '', 'mode': 'none', 'elapsed': 0, 'perf': {}}
    if mode == 'local' and not ai_engine.local.available():
        return {'ok': False, 'error': '本地模型不可用', 'answer': '本地模型文件或 llm_ask 不可用，请检查 /userdata/models/Qwen.rkllm 和 /userdata/openclaw_workspace/llm_ask', 'mode': 'local', 'elapsed': 0, 'perf': {}}
    ai_state['requests'] += 1
    start = time.time()
    result = ai_engine.ask(prompt, mode=mode)
    elapsed = time.time() - start
    if result['mode'] == 'local':
        ai_state['total_local'] += 1
        ai_state['avg_time_local'] = round(
            (ai_state['avg_time_local'] * (ai_state['total_local'] - 1) + elapsed) / ai_state['total_local'], 2
        )
    else:
        ai_state['total_cloud'] += 1
        ai_state['avg_time_cloud'] = round(
            (ai_state['avg_time_cloud'] * (ai_state['total_cloud'] - 1) + elapsed) / ai_state['total_cloud'], 2
        )
    ai_state['last_answer'] = result.get('answer', '')
    return {
        'ok': result['success'],
        'answer': result['answer'],
        'mode': result['mode'],
        'elapsed': result['elapsed'],
        'perf': result.get('perf', {}),
    }


@app.route('/ai/form_ask', methods=['POST'])
def ai_form_ask():
    prompt = (request.form.get('q') or '').strip()
    result = run_ai_prompt(prompt)
    answer = result.get('answer') or result.get('error') or ''
    safe_prompt = html.escape(prompt)
    safe_answer = html.escape(answer)
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>AI 本地推理结果</title><style>body{{margin:0;background:#0b1020;color:#e9eefc;font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;padding:28px}}.wrap{{max-width:900px;margin:0 auto}}pre{{white-space:pre-wrap;background:#0a1220;border:1px solid rgba(255,255,255,.10);border-radius:14px;padding:16px;line-height:1.65}}a,button{{display:inline-block;margin-top:14px;background:#6f96f5;color:white;text-decoration:none;border:0;border-radius:12px;padding:11px 16px;font-weight:700}}</style></head><body><div class="wrap"><h2>AI 本地推理结果</h2><div>问题：{safe_prompt}</div><div>模式：{html.escape(str(result.get('mode')))} · 耗时：{html.escape(str(result.get('elapsed')))}s</div><pre>{safe_answer}</pre><a href="/">返回控制面板</a></div></body></html>'''


@app.route('/api/ai/ask', methods=['POST'])
def ai_ask():
    data = request.get_json(force=True, silent=True) or {}
    prompt = (data.get('q') or data.get('text') or '').strip()
    mode = data.get('mode') or ai_state.get('mode', 'local')
    result = run_ai_prompt(prompt, mode=mode)
    status = 200 if result.get('ok') else 400
    return jsonify(result), status


@app.route('/api/ai/mode', methods=['POST'])
def ai_mode():
    data = request.get_json(force=True, silent=True) or {}
    mode = data.get('mode', '').strip()
    if mode not in ('local', 'cloud'):
        return jsonify({'ok': False, 'error': '模式必须是 local 或 cloud'})
    ai_engine.switch_mode(mode)
    ai_state['mode'] = mode
    return jsonify({'ok': True, 'mode': mode})


@app.route('/api/ai/config', methods=['GET', 'PUT'])
def ai_config():
    if request.method == 'PUT':
        data = request.get_json(force=True, silent=True) or {}
        updates = {}
        if 'api_key' in data:
            updates['cloud'] = updates.get('cloud', {})
            updates['cloud']['api_key'] = data['api_key']
        if 'api_url' in data:
            updates['cloud'] = updates.get('cloud', {})
            updates['cloud']['api_url'] = data['api_url']
        if 'model' in data:
            updates['cloud'] = updates.get('cloud', {})
            updates['cloud']['model'] = data['model']
        if updates:
            ai_engine.update_config(updates)
        return jsonify({'ok': True, 'config': ai_engine.get_config_safe()})
    return jsonify(ai_engine.get_config_safe())


@app.route('/api/ai/status')
def ai_status():
    return jsonify({
        'mode': ai_state['mode'],
        'status': ai_engine.get_status(),
        'stats': {
            'total_requests': ai_state['requests'],
            'total_local': ai_state['total_local'],
            'total_cloud': ai_state['total_cloud'],
            'avg_time_local': ai_state['avg_time_local'],
            'avg_time_cloud': ai_state['avg_time_cloud'],
            'local_available': ai_engine.local.available(),
            'cloud_available': ai_engine.cloud.available(),
        },
        'config_safe': ai_engine.get_config_safe(),
        'preset_questions': ai_state['preset_questions'],
    })


@app.route('/api/ai/ask_auto', methods=['POST'])
def ai_ask_auto():
    data = request.get_json(force=True, silent=True) or {}
    prompt = (data.get('q') or data.get('text') or '').strip()
    if not prompt:
        return jsonify({'ok': False, 'error': '请输入问题'})
    ai_state['requests'] += 1
    result = ai_engine.ask_auto(prompt)
    ai_state['last_answer'] = result.get('answer', '')
    if result['mode'] == 'local':
        ai_state['total_local'] += 1
    else:
        ai_state['total_cloud'] += 1
    return jsonify({
        'ok': result['success'],
        'answer': result['answer'],
        'mode': result['mode'],
        'elapsed': result['elapsed'],
        'perf': result.get('perf', {}),
    })


@app.route('/api/ai/reset_stats', methods=['POST'])
def ai_reset_stats():
    ai_state['requests'] = 0
    ai_state['total_local'] = 0
    ai_state['total_cloud'] = 0
    ai_state['avg_time_local'] = 0
    ai_state['avg_time_cloud'] = 0
    ai_state['last_answer'] = ''
    return jsonify({'ok': True, 'message': '统计数据已重置'})


# ============================================================
# 一键启动 API
# ============================================================

LAUNCH_CMDS = {
    5000: 'cd /userdata && nohup python3 media4.py > /dev/null 2>&1 &',
    5001: 'cd /userdata && nohup python3 mediapipe_mouse_control.py > /dev/null 2>&1 &',
    5002: None,
    8080: 'cd /home/linaro/.openclaw/workspace && bash start_rknn_system.sh > /dev/null 2>&1 &',
    8083: 'cd /home/linaro/.openclaw/workspace && bash start_ningning_complete.sh > /dev/null 2>&1 &',
    8088: 'cd /userdata && nohup python3 zongtitrack4.py > /dev/null 2>&1 &',
    8095: 'cd /userdata && nohup python3 tts5_web.py > /dev/null 2>&1 &',
    9880: None,
}


@app.route('/api/launch/<int:port>', methods=['POST'])
def api_launch(port):
    # 先检查是否已活着
    try:
        r = requests.get(f'http://127.0.0.1:{port}/', timeout=2)
        if r.status_code < 500:
            return jsonify({'ok': True, 'message': f'端口 {port} 已在运行', 'alive': True})
    except:
        pass

    cmd = LAUNCH_CMDS.get(port)
    if not cmd:
        # 5002/9880 用 ensure_all
        if port == 5002:
            ensure_all()
        elif port == 9880:
            ensure_gpt_sovits()
    else:
        try:
            subprocess.Popen(cmd, shell=True, stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           preexec_fn=os.setsid)
        except Exception as e:
            return jsonify({'ok': False, 'error': str(e)}), 500

    return jsonify({'ok': True, 'message': f'正在启动端口 {port}', 'alive': False})


if __name__ == '__main__':
    ensure_all()
    kill_visual_processes()
    start_live2d_gesture_camera()
    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)
