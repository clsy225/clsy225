#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TTS5 speaker-id web UI.

Features:
- build speaker gallery from uploaded reference audios
- identify uploaded query audio with unknown threshold
- compare two audios
- browse current gallery
- record query audio from browser microphone
- record reference/query audio from host ALSA microphone

This is a standalone upgrade path for tts5 speaker identification,
meant to be embedded later into the integrated page.
"""

from pathlib import Path
import json
import os
import shutil
import traceback
from datetime import datetime
import subprocess
import wave

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

from speaker_id_tool import SpeakerID, cosine
import numpy as np

APP_DIR = Path('/home/linaro/.openclaw/workspace')
DATA_DIR = APP_DIR / '.tmp' / 'openclaw-spikes' / 'tts5-webui'
REF_DIR = DATA_DIR / 'refs'
QUERY_DIR = DATA_DIR / 'queries'
GALLERY_JSON = DATA_DIR / 'speaker_gallery_web.json'
MODEL_PATH = APP_DIR / 'voxblink2_samresnet34_fp_rk3588.rknn'
ALLOWED = {'.wav', '.flac', '.mp3', '.m4a', '.ogg', '.webm'}
DEFAULT_CAPTURE_DEVICE = 'plughw:CARD=UACDemoV10,DEV=0'
FALLBACK_CAPTURE_DEVICES = [
    'plughw:CARD=UACDemoV10,DEV=0',
    'plughw:CARD=Camera,DEV=0',
    'default',
]

app = Flask(__name__)


def ensure_dirs():
    REF_DIR.mkdir(parents=True, exist_ok=True)
    QUERY_DIR.mkdir(parents=True, exist_ok=True)


def allowed(name: str) -> bool:
    return Path(name).suffix.lower() in ALLOWED


def speaker_name_from_upload(filename: str, fallback: str = 'unknown') -> str:
    stem = Path(filename).stem.strip()
    if not stem:
        return fallback
    if '_' in stem:
        return stem.split('_')[0]
    return stem


def load_gallery():
    if not GALLERY_JSON.exists():
        return {}
    return json.loads(GALLERY_JSON.read_text(encoding='utf-8'))


def save_gallery(gallery):
    GALLERY_JSON.write_text(json.dumps(gallery, ensure_ascii=False, indent=2), encoding='utf-8')


def list_ref_files():
    out = []
    for speaker_dir in sorted(REF_DIR.glob('*')):
        if not speaker_dir.is_dir():
            continue
        files = []
        for p in sorted(speaker_dir.iterdir()):
            if not p.is_file():
                continue
            files.append({
                'name': p.name,
                'path': str(p),
                'size': p.stat().st_size,
                'mtime': datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
            })
        out.append({'speaker': speaker_dir.name, 'files': files, 'count': len(files)})
    return out


def delete_ref_file(speaker: str, filename: str):
    speaker = secure_filename((speaker or '').strip())
    filename = Path(filename or '').name
    if not speaker:
        raise ValueError('missing speaker')
    if not filename:
        raise ValueError('missing filename')
    target = REF_DIR / speaker / filename
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(f'ref file not found: {speaker}/{filename}')
    target.unlink()
    speaker_dir = REF_DIR / speaker
    if speaker_dir.exists() and not any(speaker_dir.iterdir()):
        speaker_dir.rmdir()
    return {'speaker': speaker, 'filename': filename, 'deleted': True}


def delete_speaker_refs(speaker: str):
    speaker = secure_filename((speaker or '').strip())
    if not speaker:
        raise ValueError('missing speaker')
    speaker_dir = REF_DIR / speaker
    if not speaker_dir.exists() or not speaker_dir.is_dir():
        raise FileNotFoundError(f'speaker not found: {speaker}')
    count = sum(1 for p in speaker_dir.iterdir() if p.is_file())
    shutil.rmtree(speaker_dir)
    return {'speaker': speaker, 'deleted_files': count, 'deleted': True}


def gallery_needs_rebuild(gallery):
    try:
        ref_state = {
            item['speaker']: [f['name'] for f in item.get('files', [])]
            for item in list_ref_files()
        }
        gallery_state = {
            speaker: sorted(Path(it['file']).name for it in items)
            for speaker, items in (gallery or {}).items()
        }
        return ref_state != gallery_state
    except Exception:
        return False


def summarize_identify_result(result):
    top_k = result.get('top_k') or []
    best_name = top_k[0][0] if top_k else None
    best_score = top_k[0][1] if top_k else None
    result['best_match_speaker'] = best_name
    result['best_match_score'] = best_score
    result['decision_speaker'] = result.get('speaker')
    result['is_unknown'] = (result.get('speaker') == 'unknown')
    return result


def run_capture(seconds: int, out_path: Path, device: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        'arecord',
        '-D', device,
        '-f', 'S16_LE',
        '-c', '1',
        '-r', '16000',
        '-d', str(seconds),
        str(out_path),
    ]
    cp = subprocess.run(cmd, text=True, capture_output=True)
    return {
        'cmd': ' '.join(cmd),
        'returncode': cp.returncode,
        'stdout': cp.stdout,
        'stderr': cp.stderr,
    }


def probe_capture_devices():
    devices = []
    try:
        cp = subprocess.run(['arecord', '-L'], text=True, capture_output=True, check=False)
        lines = [x.rstrip() for x in cp.stdout.splitlines() if x.strip()]
        current = []
        for line in lines:
            if not line.startswith(' '):
                devices.append(line.strip())
    except Exception:
        pass
    return devices


def build_gallery_from_refs(model_path=MODEL_PATH, segments=3):
    sid = SpeakerID(str(model_path), segments)
    try:
        gallery = {}
        for speaker_dir in sorted(REF_DIR.glob('*')):
            if not speaker_dir.is_dir():
                continue
            for p in sorted(speaker_dir.iterdir()):
                if not p.is_file() or p.suffix.lower() not in ALLOWED:
                    continue
                emb = sid.embed_wav(str(p)).tolist()
                gallery.setdefault(speaker_dir.name, []).append({
                    'file': str(p),
                    'embedding': emb,
                })
        save_gallery(gallery)
        return gallery
    finally:
        sid.close()


def identify_file(query_path, threshold=0.70, margin=0.0, topk=5, segments=3, model_path=MODEL_PATH):
    gallery = load_gallery()
    if not gallery:
        raise RuntimeError('gallery is empty, please build gallery first')

    sid = SpeakerID(str(model_path), segments)
    try:
        q = sid.embed_wav(str(query_path))
        scores = []
        for speaker, items in gallery.items():
            sims = [cosine(q, np.array(it['embedding'], dtype=np.float32)) for it in items]
            score = max(sims)
            scores.append((speaker, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        best = scores[0]
        second = scores[1] if len(scores) > 1 else (None, -1.0)
        m = best[1] - second[1]
        unknown = (best[1] < threshold)
        result = {
            'query': str(query_path),
            'speaker': 'unknown' if unknown else best[0],
            'best_score': best[1],
            'second_score': second[1],
            'margin': m,
            'threshold': threshold,
            'margin_threshold': margin,
            'top_k': scores[:topk],
        }
        return summarize_identify_result(result)
    finally:
        sid.close()


@app.route('/')
def index():
    return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>TTS5 声纹建库与识别</title>
  <style>
    body { font-family: Arial, sans-serif; background:#111827; color:#e5e7eb; margin:0; padding:24px; }
    h1,h2,h3 { margin:0 0 12px; }
    .wrap { max-width:1200px; margin:0 auto; }
    .card { background:#1f2937; border:1px solid #374151; border-radius:16px; padding:18px; margin:0 0 18px; }
    .row { display:flex; gap:18px; flex-wrap:wrap; }
    .col { flex:1 1 360px; }
    input, button, select { width:100%; box-sizing:border-box; padding:10px 12px; margin:8px 0; border-radius:10px; border:1px solid #4b5563; background:#0f172a; color:#fff; }
    button { background:#2563eb; border:none; cursor:pointer; font-weight:600; }
    button.secondary { background:#374151; }
    button.danger { background:#b91c1c; }
    button.warn { background:#92400e; }
    button:disabled { opacity:.7; cursor:not-allowed; }
    pre { white-space:pre-wrap; word-break:break-word; background:#0b1220; padding:12px; border-radius:12px; border:1px solid #374151; }
    .muted { color:#9ca3af; font-size:14px; }
    .ok { color:#34d399; }
    .warn-text { color:#fbbf24; }
    label { display:block; margin-top:8px; color:#cbd5e1; }
    .btn-row { display:flex; gap:10px; }
    .btn-row button { flex:1; }
    .toast {
      position: fixed;
      right: 20px;
      bottom: 20px;
      min-width: 260px;
      max-width: 420px;
      background: #111827;
      color: #fff;
      border: 1px solid #4b5563;
      border-radius: 12px;
      padding: 12px 14px;
      box-shadow: 0 12px 30px rgba(0,0,0,.35);
      z-index: 9999;
      opacity: 0;
      transform: translateY(8px);
      transition: all .2s ease;
      pointer-events: none;
    }
    .toast.show { opacity: 1; transform: translateY(0); }
    .toast.ok { border-color: #10b981; }
    .toast.warn { border-color: #f59e0b; }
    .toast.err { border-color: #ef4444; }
  </style>
</head>
<body>
<div class="wrap">
  <h1>🎤 TTS5 声纹建库与识别</h1>
  <p class="muted">支持：文件上传、浏览器麦克风录音、板子本机 ALSA 麦克风录音。</p>

  <div class="card">
    <h2>状态</h2>
    <div id="status">加载中...</div>
  </div>

  <div class="row">
    <div class="col card">
      <h2>1) 上传参考音频</h2>
      <label>说话人名称</label>
      <input id="speakerName" placeholder="例如：ningning / murasame" oninput="syncSpeakerNames('speakerName')" />
      <label>参考音频文件（可多选）</label>
      <input id="refFiles" type="file" multiple accept=".wav,.flac,.mp3,.m4a,.ogg,.webm" />
      <button id="uploadRefBtn" onclick="uploadRefs()">上传到参考库</button>
      <pre id="refResult">等待上传...</pre>
    </div>

    <div class="col card">
      <h2>2) 浏览器麦克风录制参考音频</h2>
      <label>说话人名称</label>
      <input id="browserSpeakerName" placeholder="例如：ningning" oninput="syncSpeakerNames('browserSpeakerName')" />
      <div class="btn-row">
        <button class="warn" id="browserRefStartBtn" onclick="startBrowserRecord()">开始录音</button>
        <button class="secondary" id="browserRefStopBtn" onclick="stopBrowserRecord()">停止录音</button>
      </div>
      <audio id="browserRefPreview" controls style="width:100%; margin-top:10px;"></audio>
      <button id="uploadBrowserRefBtn" onclick="uploadBrowserRef()">把浏览器录音存为参考音频</button>
      <pre id="browserRefResult">等待录音...</pre>
    </div>
  </div>

  <div class="row">
    <div class="col card">
      <h2>3) 板子本机麦克风录参考音频</h2>
      <label>说话人名称</label>
      <input id="hostSpeakerName" placeholder="例如：ningning" oninput="syncSpeakerNames('hostSpeakerName')" />
      <label>设备</label>
      <select id="captureDevice"></select>
      <label>录音秒数</label>
      <input id="captureSeconds" type="number" value="4" min="1" max="20" />
      <button id="hostRefBtn" onclick="recordHostRef()">用板子麦克风录参考音频</button>
      <pre id="hostRefResult">等待录音...</pre>
    </div>

    <div class="col card">
      <h2>4) 建库</h2>
      <label>分段数 segments</label>
      <input id="segments" type="number" value="3" min="1" max="8" />
      <button id="buildGalleryBtn" onclick="buildGallery()">从参考库生成 gallery</button>
      <pre id="buildResult">等待建库...</pre>
    </div>
  </div>

  <div class="row">
    <div class="col card">
      <h2>5) 文件上传识别</h2>
      <label>查询音频</label>
      <input id="queryFile" type="file" accept=".wav,.flac,.mp3,.m4a,.ogg,.webm" />
      <label>threshold</label>
      <input id="threshold" type="number" value="0.70" step="0.01" />
      <label>margin</label>
      <input id="margin" type="number" value="0.00" step="0.01" />
      <label>topk</label>
      <input id="topk" type="number" value="5" min="1" max="20" />
      <button id="identifyFileBtn" onclick="identifyAudio()">开始识别</button>
      <pre id="identifyResult">等待识别...</pre>
    </div>

    <div class="col card">
      <h2>6) 浏览器麦克风直接识别</h2>
      <div class="btn-row">
        <button class="warn" id="browserQueryStartBtn" onclick="startBrowserQueryRecord()">开始录音</button>
        <button class="secondary" id="browserQueryStopBtn" onclick="stopBrowserQueryRecord()">停止录音</button>
      </div>
      <audio id="browserQueryPreview" controls style="width:100%; margin-top:10px;"></audio>
      <button id="identifyBrowserBtn" onclick="identifyBrowserQuery()">识别浏览器录音</button>
      <pre id="browserQueryResult">等待录音...</pre>
    </div>
  </div>

  <div class="row">
    <div class="col card">
      <h2>7) 板子本机麦克风直接识别</h2>
      <label>设备</label>
      <select id="queryCaptureDevice"></select>
      <label>录音秒数</label>
      <input id="queryCaptureSeconds" type="number" value="4" min="1" max="20" />
      <button id="hostQueryBtn" onclick="recordHostQueryAndIdentify()">录音并识别</button>
      <pre id="hostQueryResult">等待识别...</pre>
    </div>

    <div class="col card">
      <h2>8) 参考库管理</h2>
      <label>删除某个人</label>
      <select id="deleteSpeakerSelect"></select>
      <div class="btn-row">
        <button class="secondary" onclick="refreshStatus()">查看当前库</button>
        <button class="danger" id="deleteSpeakerBtn" onclick="deleteSpeakerRefs()">删除这个人物</button>
      </div>
      <label>删除某条音频</label>
      <select id="deleteSpeakerFileSelect"></select>
      <button class="danger" id="deleteFileBtn" onclick="deleteSelectedFile()">删除这条音频</button>
      <button class="danger" onclick="clearRefs()">清空参考音频与 gallery</button>
      <pre id="galleryView">等待加载...</pre>
    </div>
  </div>
</div>

<div id="toast" class="toast"></div>

<script>
let refRecorder = null;
let refChunks = [];
let refBlob = null;
let queryRecorder = null;
let queryChunks = [];
let queryBlob = null;

function pickSpeakerName() {
  const ids = ['hostSpeakerName', 'browserSpeakerName', 'speakerName'];
  for (const id of ids) {
    const el = document.getElementById(id);
    if (el && el.value && el.value.trim()) return el.value.trim();
  }
  return '';
}

function syncSpeakerNames(sourceId) {
  const src = document.getElementById(sourceId);
  if (!src) return;
  const val = src.value || '';
  for (const id of ['speakerName', 'browserSpeakerName', 'hostSpeakerName']) {
    if (id === sourceId) continue;
    const el = document.getElementById(id);
    if (el && !el.value.trim()) el.value = val;
  }
}

function showToast(message, type='ok', timeout=2600) {
  const el = document.getElementById('toast');
  if (!el) return;
  el.textContent = message;
  el.className = `toast ${type} show`;
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => {
    el.className = `toast ${type}`;
  }, timeout);
}

function setBtnBusy(id, busy, busyText='处理中...') {
  const btn = document.getElementById(id);
  if (!btn) return;
  if (!btn.dataset.originalText) btn.dataset.originalText = btn.textContent;
  btn.disabled = !!busy;
  btn.textContent = busy ? busyText : btn.dataset.originalText;
}

function renderLibraryStatus(data) {
  const refs = data.refs_detail || [];
  document.getElementById('galleryView').textContent = JSON.stringify(refs, null, 2);

  const speakerSel = document.getElementById('deleteSpeakerSelect');
  const fileSel = document.getElementById('deleteSpeakerFileSelect');
  if (!speakerSel || !fileSel) return;

  const currentSpeaker = speakerSel.value;
  speakerSel.innerHTML = '';
  for (const item of refs) {
    const opt = document.createElement('option');
    opt.value = item.speaker;
    opt.textContent = `${item.speaker} (${item.count}条)`;
    speakerSel.appendChild(opt);
  }
  if (currentSpeaker && refs.some(x => x.speaker === currentSpeaker)) {
    speakerSel.value = currentSpeaker;
  }
  refreshFileOptions();
}

function refreshFileOptions() {
  const speakerSel = document.getElementById('deleteSpeakerSelect');
  const fileSel = document.getElementById('deleteSpeakerFileSelect');
  if (!speakerSel || !fileSel) return;
  const refs = window._lastRefsDetail || [];
  const current = refs.find(x => x.speaker === speakerSel.value) || refs[0];
  fileSel.innerHTML = '';
  if (!current) return;
  for (const file of (current.files || [])) {
    const opt = document.createElement('option');
    opt.value = `${current.speaker}::${file.name}`;
    const kb = (file.size / 1024).toFixed(1);
    opt.textContent = `${file.name} (${kb} KB)`;
    fileSel.appendChild(opt);
  }
}

async function jsonFetch(url, opts={}) {
  const r = await fetch(url, opts);
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || JSON.stringify(data));
  return data;
}

async function refreshStatus() {
  try {
    const data = await jsonFetch('/api/status');
    document.getElementById('status').innerHTML = `
      <div>模型: <span class="ok">${data.model_exists ? '已找到' : '未找到'}</span></div>
      <div>默认采集设备: <span class="ok">${data.default_capture_device}</span></div>
      <div>参考说话人数: <span class="ok">${data.ref_speakers}</span></div>
      <div>参考音频总数: <span class="ok">${data.ref_files}</span></div>
      <div>gallery 说话人数: <span class="ok">${data.gallery_speakers}</span></div>
      <div>gallery 条目数: <span class="ok">${data.gallery_items}</span></div>
      <div>gallery 状态: <span class="${data.gallery_stale ? 'warn-text' : 'ok'}">${data.gallery_stale ? '已过期，需要重新建库' : '已同步'}</span></div>
      <div class="muted">gallery 文件: ${data.gallery_path}</div>
    `;
    window._lastRefsDetail = data.refs_detail || [];
    renderLibraryStatus(data);

    const sels = [document.getElementById('captureDevice'), document.getElementById('queryCaptureDevice')];
    for (const sel of sels) {
      sel.innerHTML = '';
      for (const dev of data.capture_devices) {
        const opt = document.createElement('option');
        opt.value = dev;
        opt.textContent = dev;
        if (dev === data.default_capture_device) opt.selected = true;
        sel.appendChild(opt);
      }
    }
  } catch (e) {
    document.getElementById('status').textContent = '状态读取失败: ' + e.message;
  }
}

async function uploadRefs() {
  const files = document.getElementById('refFiles').files;
  const speaker = pickSpeakerName();
  if (!files.length) return alert('先选参考音频');
  const fd = new FormData();
  if (speaker) fd.append('speaker', speaker);
  for (const f of files) fd.append('files', f);
  try {
    showToast('开始上传参考音频...', 'warn', 1600);
    setBtnBusy('uploadRefBtn', true, '上传中...');
    const data = await jsonFetch('/api/upload_refs', { method:'POST', body: fd });
    document.getElementById('refResult').textContent = JSON.stringify(data, null, 2);
    showToast(`上传完成：新增 ${data.count} 条参考音频`, 'ok', 3000);
    refreshStatus();
  } catch (e) {
    document.getElementById('refResult').textContent = '上传失败: ' + e.message;
    showToast('上传失败：' + e.message, 'err', 4200);
  } finally {
    setBtnBusy('uploadRefBtn', false);
  }
}

async function buildGallery() {
  try {
    showToast('开始建库，请稍等...', 'warn', 1800);
    setBtnBusy('buildGalleryBtn', true, '建库中...');
    const body = JSON.stringify({ segments: parseInt(document.getElementById('segments').value || '3', 10) });
    const data = await jsonFetch('/api/build_gallery', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body
    });
    document.getElementById('buildResult').textContent = JSON.stringify(data, null, 2);
    showToast(`建库完成：${data.count_speakers} 个说话人 / ${data.count_items} 条参考音频`, 'ok', 3200);
    refreshStatus();
  } catch (e) {
    document.getElementById('buildResult').textContent = '建库失败: ' + e.message;
    showToast('建库失败：' + e.message, 'err', 4200);
  } finally {
    setBtnBusy('buildGalleryBtn', false);
  }
}

async function identifyAudio() {
  const file = document.getElementById('queryFile').files[0];
  if (!file) return alert('先选查询音频');
  const fd = new FormData();
  fd.append('file', file);
  fd.append('threshold', document.getElementById('threshold').value);
  fd.append('margin', document.getElementById('margin').value);
  fd.append('topk', document.getElementById('topk').value);
  fd.append('segments', document.getElementById('segments').value);
  try {
    showToast('开始识别上传音频...', 'warn', 1800);
    setBtnBusy('identifyFileBtn', true, '识别中...');
    const data = await jsonFetch('/api/identify', { method:'POST', body: fd });
    document.getElementById('identifyResult').textContent = JSON.stringify(data, null, 2);
    showToast(`识别完成：${data.speaker}（${(data.best_score || 0).toFixed(3)}）`, 'ok', 3600);
  } catch (e) {
    document.getElementById('identifyResult').textContent = '识别失败: ' + e.message;
    showToast('识别失败：' + e.message, 'err', 4200);
  } finally {
    setBtnBusy('identifyFileBtn', false);
  }
}

async function deleteSpeakerRefs() {
  const speaker = document.getElementById('deleteSpeakerSelect').value;
  if (!speaker) return alert('先选一个人物');
  if (!confirm(`确认删除人物 ${speaker} 的全部参考音频吗？`)) return;
  try {
    setBtnBusy('deleteSpeakerBtn', true, '删除中...');
    showToast(`正在删除人物 ${speaker} ...`, 'warn', 1800);
    const data = await jsonFetch('/api/delete_speaker', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ speaker })
    });
    showToast(`已删除人物 ${data.speaker}，共 ${data.deleted_files} 条`, 'ok', 3200);
    await refreshStatus();
  } catch (e) {
    showToast('删除人物失败：' + e.message, 'err', 4200);
  } finally {
    setBtnBusy('deleteSpeakerBtn', false);
  }
}

async function deleteSelectedFile() {
  const value = document.getElementById('deleteSpeakerFileSelect').value;
  if (!value) return alert('先选一条音频');
  const [speaker, filename] = value.split('::');
  if (!speaker || !filename) return alert('音频选择无效');
  if (!confirm(`确认删除 ${speaker} / ${filename} 吗？`)) return;
  try {
    setBtnBusy('deleteFileBtn', true, '删除中...');
    showToast(`正在删除音频 ${filename} ...`, 'warn', 1800);
    const data = await jsonFetch('/api/delete_ref_file', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ speaker, filename })
    });
    showToast(`已删除音频：${data.filename}`, 'ok', 3200);
    await refreshStatus();
  } catch (e) {
    showToast('删除音频失败：' + e.message, 'err', 4200);
  } finally {
    setBtnBusy('deleteFileBtn', false);
  }
}

async function clearRefs() {
  if (!confirm('确认清空参考音频和 gallery？')) return;
  try {
    const data = await jsonFetch('/api/clear_refs', { method:'POST' });
    document.getElementById('galleryView').textContent = JSON.stringify(data, null, 2);
    refreshStatus();
  } catch (e) {
    document.getElementById('galleryView').textContent = '清空失败: ' + e.message;
  }
}

async function startBrowserRecord() {
  try {
    const getUserMediaFn = (
      navigator.mediaDevices && navigator.mediaDevices.getUserMedia
    ) ? (opts) => navigator.mediaDevices.getUserMedia(opts)
      : (navigator.getUserMedia
        || navigator.webkitGetUserMedia
        || navigator.mozGetUserMedia
        || navigator.msGetUserMedia)
        ? (opts) => new Promise((resolve, reject) => {
            (navigator.getUserMedia
              || navigator.webkitGetUserMedia
              || navigator.mozGetUserMedia
              || navigator.msGetUserMedia).call(navigator, opts, resolve, reject);
          })
        : null;

    if (!getUserMediaFn) {
      throw new Error('当前浏览器/容器不支持 getUserMedia。请改用“板子本机麦克风录音”或换 Chrome/Edge/Firefox 正常浏览器打开。');
    }
    if (typeof MediaRecorder === 'undefined') {
      throw new Error('当前浏览器/容器不支持 MediaRecorder。请改用“板子本机麦克风录音”。');
    }

    const stream = await getUserMediaFn({ audio: true });
    refChunks = [];
    refBlob = null;
    refRecorder = new MediaRecorder(stream);
    refRecorder.ondataavailable = e => { if (e.data.size > 0) refChunks.push(e.data); };
    refRecorder.onstop = () => {
      refBlob = new Blob(refChunks, { type: 'audio/webm' });
      document.getElementById('browserRefPreview').src = URL.createObjectURL(refBlob);
      document.getElementById('browserRefResult').textContent = '录音完成，可上传到参考库';
      showToast('浏览器录音结束，可以上传到参考库', 'ok', 2600);
      setBtnBusy('browserRefStartBtn', false);
      stream.getTracks().forEach(t => t.stop());
    };
    refRecorder.start();
    setBtnBusy('browserRefStartBtn', true, '录音中...');
    showToast('浏览器录音已开始，请说话...', 'warn', 2200);
    document.getElementById('browserRefResult').textContent = '录音中...';
  } catch (e) {
    setBtnBusy('browserRefStartBtn', false);
    document.getElementById('browserRefResult').textContent = '无法打开浏览器麦克风: ' + e.message;
    showToast('浏览器麦克风不可用：' + e.message, 'err', 4200);
  }
}

function stopBrowserRecord() {
  if (refRecorder && refRecorder.state !== 'inactive') {
    showToast('正在结束浏览器录音...', 'warn', 1400);
    refRecorder.stop();
  }
}

async function uploadBrowserRef() {
  const speaker = pickSpeakerName();
  if (!speaker) return alert('先填写说话人名称');
  if (!refBlob) return alert('先录一段音');
  const fd = new FormData();
  fd.append('speaker', speaker);
  fd.append('files', new File([refBlob], `${speaker}_browser.webm`, { type: 'audio/webm' }));
  try {
    showToast('正在上传浏览器录音到参考库...', 'warn', 1800);
    setBtnBusy('uploadBrowserRefBtn', true, '上传中...');
    const data = await jsonFetch('/api/upload_refs', { method:'POST', body: fd });
    document.getElementById('browserRefResult').textContent = JSON.stringify(data, null, 2);
    showToast('浏览器录音已保存到参考库', 'ok', 2800);
    refreshStatus();
  } catch (e) {
    document.getElementById('browserRefResult').textContent = '上传浏览器录音失败: ' + e.message;
    showToast('上传浏览器录音失败：' + e.message, 'err', 4200);
  } finally {
    setBtnBusy('uploadBrowserRefBtn', false);
  }
}

async function startBrowserQueryRecord() {
  try {
    const getUserMediaFn = (
      navigator.mediaDevices && navigator.mediaDevices.getUserMedia
    ) ? (opts) => navigator.mediaDevices.getUserMedia(opts)
      : (navigator.getUserMedia
        || navigator.webkitGetUserMedia
        || navigator.mozGetUserMedia
        || navigator.msGetUserMedia)
        ? (opts) => new Promise((resolve, reject) => {
            (navigator.getUserMedia
              || navigator.webkitGetUserMedia
              || navigator.mozGetUserMedia
              || navigator.msGetUserMedia).call(navigator, opts, resolve, reject);
          })
        : null;

    if (!getUserMediaFn) {
      throw new Error('当前浏览器/容器不支持 getUserMedia。请改用“板子本机麦克风录音”或换 Chrome/Edge/Firefox 正常浏览器打开。');
    }
    if (typeof MediaRecorder === 'undefined') {
      throw new Error('当前浏览器/容器不支持 MediaRecorder。请改用“板子本机麦克风录音”。');
    }

    const stream = await getUserMediaFn({ audio: true });
    queryChunks = [];
    queryBlob = null;
    queryRecorder = new MediaRecorder(stream);
    queryRecorder.ondataavailable = e => { if (e.data.size > 0) queryChunks.push(e.data); };
    queryRecorder.onstop = () => {
      queryBlob = new Blob(queryChunks, { type: 'audio/webm' });
      document.getElementById('browserQueryPreview').src = URL.createObjectURL(queryBlob);
      document.getElementById('browserQueryResult').textContent = '录音完成，可直接识别';
      showToast('浏览器查询录音结束，可以开始识别', 'ok', 2600);
      setBtnBusy('browserQueryStartBtn', false);
      stream.getTracks().forEach(t => t.stop());
    };
    queryRecorder.start();
    setBtnBusy('browserQueryStartBtn', true, '录音中...');
    showToast('浏览器查询录音已开始，请说话...', 'warn', 2200);
    document.getElementById('browserQueryResult').textContent = '录音中...';
  } catch (e) {
    setBtnBusy('browserQueryStartBtn', false);
    document.getElementById('browserQueryResult').textContent = '无法打开浏览器麦克风: ' + e.message;
    showToast('浏览器麦克风不可用：' + e.message, 'err', 4200);
  }
}

function stopBrowserQueryRecord() {
  if (queryRecorder && queryRecorder.state !== 'inactive') {
    showToast('正在结束浏览器查询录音...', 'warn', 1400);
    queryRecorder.stop();
  }
}

async function identifyBrowserQuery() {
  if (!queryBlob) return alert('先录一段查询音频');
  const fd = new FormData();
  fd.append('file', new File([queryBlob], 'browser_query.webm', { type: 'audio/webm' }));
  fd.append('threshold', document.getElementById('threshold').value);
  fd.append('margin', document.getElementById('margin').value);
  fd.append('topk', document.getElementById('topk').value);
  fd.append('segments', document.getElementById('segments').value);
  try {
    showToast('开始识别浏览器录音...', 'warn', 1800);
    setBtnBusy('identifyBrowserBtn', true, '识别中...');
    const data = await jsonFetch('/api/identify', { method:'POST', body: fd });
    document.getElementById('browserQueryResult').textContent = JSON.stringify(data, null, 2);
    showToast(`浏览器录音识别完成：${data.speaker}（${(data.best_score || 0).toFixed(3)}）`, 'ok', 3600);
  } catch (e) {
    document.getElementById('browserQueryResult').textContent = '浏览器录音识别失败: ' + e.message;
    showToast('浏览器录音识别失败：' + e.message, 'err', 4200);
  } finally {
    setBtnBusy('identifyBrowserBtn', false);
  }
}

async function recordHostRef() {
  const speaker = pickSpeakerName();
  if (!speaker) {
    alert('先填写说话人名称，再录参考音频');
    return;
  }
  try {
    showToast('板子本机开始录参考音频，请对着麦克风说话...', 'warn', 2200);
    setBtnBusy('hostRefBtn', true, '录音中...');
    const data = await jsonFetch('/api/record_ref_host', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        speaker,
        device: document.getElementById('captureDevice').value,
        seconds: parseInt(document.getElementById('captureSeconds').value || '4', 10),
      })
    });
    document.getElementById('hostRefResult').textContent = JSON.stringify(data, null, 2);
    showToast(`参考录音完成：${data.speaker}，已保存 1 条`, 'ok', 3200);
    refreshStatus();
  } catch (e) {
    document.getElementById('hostRefResult').textContent = '板子本机录参考失败: ' + e.message;
    showToast('板子本机录参考失败：' + e.message, 'err', 4200);
  } finally {
    setBtnBusy('hostRefBtn', false);
  }
}

async function recordHostQueryAndIdentify() {
  try {
    showToast('板子本机开始录音并识别，请对着麦克风说话...', 'warn', 2200);
    setBtnBusy('hostQueryBtn', true, '录音并识别中...');
    const data = await jsonFetch('/api/record_query_host_identify', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        device: document.getElementById('queryCaptureDevice').value,
        seconds: parseInt(document.getElementById('queryCaptureSeconds').value || '4', 10),
        threshold: parseFloat(document.getElementById('threshold').value || '0.70'),
        margin: parseFloat(document.getElementById('margin').value || '0.00'),
        topk: parseInt(document.getElementById('topk').value || '5', 10),
        segments: parseInt(document.getElementById('segments').value || '3', 10),
      })
    });
    document.getElementById('hostQueryResult').textContent = JSON.stringify(data, null, 2);
    showToast(`识别完成：${data.speaker}（${(data.best_score || 0).toFixed(3)}）`, 'ok', 3600);
  } catch (e) {
    document.getElementById('hostQueryResult').textContent = '板子本机录音识别失败: ' + e.message;
    showToast('板子本机录音识别失败：' + e.message, 'err', 4200);
  } finally {
    setBtnBusy('hostQueryBtn', false);
  }
}

refreshStatus();
const deleteSpeakerSelect = document.getElementById('deleteSpeakerSelect');
if (deleteSpeakerSelect) deleteSpeakerSelect.addEventListener('change', refreshFileOptions);
</script>
</body>
</html>
"""


@app.route('/api/status')
def api_status():
    ensure_dirs()
    refs = list_ref_files()
    gallery = load_gallery()
    stale = gallery_needs_rebuild(gallery)
    return jsonify({
        'ok': True,
        'model_exists': MODEL_PATH.exists(),
        'gallery_path': str(GALLERY_JSON),
        'default_capture_device': DEFAULT_CAPTURE_DEVICE,
        'capture_devices': probe_capture_devices(),
        'ref_speakers': len(refs),
        'ref_files': sum(x['count'] for x in refs),
        'gallery_speakers': len(gallery),
        'gallery_items': sum(len(v) for v in gallery.values()),
        'gallery_stale': stale,
        'refs_detail': refs,
    })


@app.route('/api/upload_refs', methods=['POST'])
def api_upload_refs():
    ensure_dirs()
    files = request.files.getlist('files')
    if not files:
        return jsonify({'error': 'no files uploaded'}), 400

    explicit_speaker = (request.form.get('speaker') or '').strip()
    saved = []
    for f in files:
        if not f or not f.filename:
            continue
        if not allowed(f.filename):
            continue
        speaker = explicit_speaker or speaker_name_from_upload(f.filename)
        speaker_dir = REF_DIR / secure_filename(speaker)
        speaker_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')
        filename = secure_filename(Path(f.filename).name)
        out = speaker_dir / f'{ts}_{filename}'
        f.save(str(out))
        saved.append({'speaker': speaker, 'file': str(out)})

    return jsonify({'ok': True, 'saved': saved, 'count': len(saved)})


@app.route('/api/record_ref_host', methods=['POST'])
def api_record_ref_host():
    ensure_dirs()
    data = request.get_json(silent=True) or {}
    speaker = secure_filename((data.get('speaker') or '').strip())
    if not speaker:
        return jsonify({'error': 'missing speaker: 请先填写说话人名称，再录参考音频'}), 400
    seconds = max(1, min(20, int(data.get('seconds', 4))))
    device = (data.get('device') or DEFAULT_CAPTURE_DEVICE).strip()

    speaker_dir = REF_DIR / speaker
    speaker_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')
    out = speaker_dir / f'{ts}_host.wav'
    rec = run_capture(seconds, out, device)
    if rec['returncode'] != 0:
        return jsonify({'error': 'arecord failed', 'record': rec}), 500
    return jsonify({'ok': True, 'speaker': speaker, 'file': str(out), 'seconds': seconds, 'device': device, 'record': rec})


@app.route('/api/build_gallery', methods=['POST'])
def api_build_gallery():
    ensure_dirs()
    data = request.get_json(silent=True) or {}
    segments = int(data.get('segments', 3))
    if not MODEL_PATH.exists():
        return jsonify({'error': f'model not found: {MODEL_PATH}'}), 400
    gallery = build_gallery_from_refs(MODEL_PATH, segments)
    return jsonify({
        'ok': True,
        'gallery_path': str(GALLERY_JSON),
        'speakers': {k: len(v) for k, v in gallery.items()},
        'count_speakers': len(gallery),
        'count_items': sum(len(v) for v in gallery.values()),
        'segments': segments,
    })


@app.route('/api/identify', methods=['POST'])
def api_identify():
    ensure_dirs()
    f = request.files.get('file')
    if not f or not f.filename:
        return jsonify({'error': 'missing query file'}), 400
    if not allowed(f.filename):
        return jsonify({'error': 'unsupported file type'}), 400

    threshold = float(request.form.get('threshold', 0.70))
    margin = float(request.form.get('margin', 0.0))
    topk = int(request.form.get('topk', 5))
    segments = int(request.form.get('segments', 3))

    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')
    out = QUERY_DIR / f'{ts}_{secure_filename(f.filename)}'
    f.save(str(out))

    result = identify_file(out, threshold=threshold, margin=margin, topk=topk, segments=segments)
    return jsonify({'ok': True, **result})


@app.route('/api/record_query_host_identify', methods=['POST'])
def api_record_query_host_identify():
    ensure_dirs()
    data = request.get_json(silent=True) or {}
    seconds = max(1, min(20, int(data.get('seconds', 4))))
    device = (data.get('device') or DEFAULT_CAPTURE_DEVICE).strip()
    threshold = float(data.get('threshold', 0.70))
    margin = float(data.get('margin', 0.0))
    topk = int(data.get('topk', 5))
    segments = int(data.get('segments', 3))

    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%S%f')
    out = QUERY_DIR / f'{ts}_host_query.wav'
    rec = run_capture(seconds, out, device)
    if rec['returncode'] != 0:
        return jsonify({'error': 'arecord failed', 'record': rec}), 500
    result = identify_file(out, threshold=threshold, margin=margin, topk=topk, segments=segments)
    return jsonify({'ok': True, 'record': rec, **result})


@app.route('/api/delete_ref_file', methods=['POST'])
def api_delete_ref_file():
    ensure_dirs()
    data = request.get_json(silent=True) or {}
    try:
        result = delete_ref_file(data.get('speaker'), data.get('filename'))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    return jsonify({'ok': True, **result})


@app.route('/api/delete_speaker', methods=['POST'])
def api_delete_speaker():
    ensure_dirs()
    data = request.get_json(silent=True) or {}
    try:
        result = delete_speaker_refs(data.get('speaker'))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 404
    return jsonify({'ok': True, **result})


@app.route('/api/clear_refs', methods=['POST'])
def api_clear_refs():
    if REF_DIR.exists():
        shutil.rmtree(REF_DIR)
    if QUERY_DIR.exists():
        shutil.rmtree(QUERY_DIR)
    if GALLERY_JSON.exists():
        GALLERY_JSON.unlink()
    ensure_dirs()
    return jsonify({'ok': True, 'message': 'refs and gallery cleared'})


@app.route('/files/<path:subpath>')
def files(subpath):
    return send_from_directory(DATA_DIR, subpath, as_attachment=False)


@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({
        'ok': False,
        'error': str(e),
        'traceback': traceback.format_exc(),
    }), 500


if __name__ == '__main__':
    ensure_dirs()
    print('🌐 TTS5 声纹页面: http://0.0.0.0:8095')
    app.run(host='0.0.0.0', port=8095, debug=False, threaded=True)
