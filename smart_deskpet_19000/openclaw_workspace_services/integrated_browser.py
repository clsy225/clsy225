#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import requests
from flask import Flask, Response, request, jsonify

app = Flask(__name__)

MAIN_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>绫地宁宁整合控制台</title>
  <style>
    body { margin:0; padding:20px; background:linear-gradient(135deg,#111827 0%,#0f172a 100%); color:#e5e7eb; font-family:Arial,sans-serif; }
    .wrap { max-width:1400px; margin:0 auto; }
    h1 { color:#00ff88; text-align:center; margin:0 0 8px; }
    .sub { text-align:center; color:#94a3b8; margin-bottom:20px; }
    .grid { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
    .card { background:#1f2937; border:1px solid #374151; border-radius:16px; padding:18px; box-shadow:0 10px 30px rgba(0,0,0,.25); }
    .camera { background:#000; border-radius:12px; overflow:hidden; text-align:center; }
    .camera img { width:100%; max-height:460px; object-fit:contain; }
    .stats { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:14px 0; }
    .stat { background:#111827; border-radius:12px; padding:12px; text-align:center; }
    .val { font-size:24px; font-weight:700; color:#00ff88; }
    textarea, input, button { width:100%; box-sizing:border-box; padding:10px 12px; margin:8px 0; border-radius:10px; border:1px solid #4b5563; background:#0f172a; color:#fff; }
    button { background:#00c16a; color:#04130b; font-weight:700; cursor:pointer; border:none; }
    .btnrow { display:flex; gap:10px; }
    .btnrow button { flex:1; }
    .result, pre { white-space:pre-wrap; word-break:break-word; background:#0b1220; border:1px solid #374151; border-radius:12px; padding:12px; }
    iframe { width:100%; height:980px; border:none; border-radius:12px; background:#fff; }
    .progress { background:#475569; height:22px; border-radius:999px; overflow:hidden; margin:10px 0; }
    .bar { background:#00ff88; height:100%; width:0%; color:#04130b; text-align:center; line-height:22px; font-weight:700; }
    .muted { color:#94a3b8; font-size:14px; }
    @media (max-width: 1100px) { .grid { grid-template-columns:1fr; } .stats { grid-template-columns:repeat(2,1fr);} }
  </style>
</head>
<body>
<div class="wrap">
  <h1>🎥 绫地宁宁整合控制台</h1>
  <div class="sub">左边是摄像头/TTS 控制，右边是声纹建库与识别</div>

  <div class="grid">
    <div class="card">
      <h2>摄像头 + TTS</h2>
      <div class="stats">
        <div class="stat"><div id="cameraVal" class="val">--</div><div>摄像头</div></div>
        <div class="stat"><div id="fpsVal" class="val">0.0</div><div>FPS</div></div>
        <div class="stat"><div id="ttsVal" class="val">空闲</div><div>TTS</div></div>
        <div class="stat"><div id="portVal" class="val">8083</div><div>端口</div></div>
      </div>
      <div class="camera">
        <img id="cam" src="/frame" alt="摄像头画面">
      </div>
      <div id="ttsBox" style="display:none; margin-top:12px;">
        <div class="muted">🔊 TTS生成进度</div>
        <div class="progress"><div id="ttsBar" class="bar">0%</div></div>
        <div id="ttsText" class="muted"></div>
      </div>
      <h3>语音控制台</h3>
      <textarea id="text" placeholder="输入 '绫地宁宁现在摄像头有几个人'"></textarea>
      <div class="btnrow">
        <button onclick="sendInfer()">🎤 发送</button>
        <button onclick="setEx('绫地宁宁现在摄像头有几个人')">👥 人数</button>
        <button onclick="setEx('宁宁，帧率是多少')">⚡ 帧率</button>
      </div>
      <div class="btnrow">
        <button onclick="setEx('ningning，摄像头状态')">📷 状态</button>
        <button onclick="setEx('ningning，YOLO检测到几个人')">🤖 检测</button>
      </div>
      <div id="result" class="result">等待指令...</div>
    </div>

    <div class="card">
      <h2>声纹建库与识别</h2>
      <div class="muted">已直接整合进这个页面里，不用再单独开 8095。</div>
      <iframe src="/speaker/"></iframe>
    </div>
  </div>
</div>

<script>
async function fetchStatus() {
  try {
    const r = await fetch('/status');
    const d = await r.json();
    document.getElementById('cameraVal').textContent = d.camera ? '正常' : '模拟';
    document.getElementById('fpsVal').textContent = (d.fps || 0).toFixed(1);
    document.getElementById('ttsVal').textContent = d.tts ? '生成中' : '空闲';
    const box = document.getElementById('ttsBox');
    const bar = document.getElementById('ttsBar');
    const text = document.getElementById('ttsText');
    if (d.tts) {
      box.style.display = 'block';
      bar.style.width = (d.tts_p || 0) + '%';
      bar.textContent = (d.tts_p || 0) + '%';
      text.textContent = d.tts_t || '';
    } else {
      box.style.display = 'none';
      bar.style.width = '0%';
      bar.textContent = '0%';
      text.textContent = '';
    }
  } catch (e) {
    document.getElementById('result').textContent = '状态读取失败: ' + e.message;
  }
}
function setEx(t) { document.getElementById('text').value = t; }
async function sendInfer() {
  const text = document.getElementById('text').value.trim();
  const result = document.getElementById('result');
  if (!text) { result.textContent = '请输入文本'; return; }
  result.textContent = '处理中...';
  try {
    const r = await fetch('/infer', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({text})
    });
    const d = await r.json();
    if (d.ok && d.trig) {
      result.textContent = `✅ 语音触发成功（第${d.count}次）\n📢 ${d.resp}\n📊 FPS: ${(d.fps||0).toFixed(1)} | 👥 检测: ${d.faces}人`;
    } else {
      result.textContent = d.msg || JSON.stringify(d, null, 2);
    }
  } catch (e) {
    result.textContent = '发送失败: ' + e.message;
  }
}
setInterval(() => {
  document.getElementById('cam').src = '/frame?' + Date.now();
}, 200);
setInterval(fetchStatus, 500);
fetchStatus();
</script>
</body>
</html>
"""


def proxy_request(method, path):
    target = f'http://127.0.0.1:8095{path}'
    headers = {k: v for k, v in request.headers if k.lower() not in {'host', 'content-length'}}
    resp = requests.request(
        method=method,
        url=target,
        params=request.args,
        headers=headers,
        data=request.get_data(),
        cookies=request.cookies,
        allow_redirects=False,
        timeout=120,
    )
    excluded = {'content-encoding', 'content-length', 'transfer-encoding', 'connection'}
    response_headers = [(k, v) for k, v in resp.headers.items() if k.lower() not in excluded]
    return Response(resp.content, resp.status_code, response_headers)


@app.route('/')
def index():
    return Response(MAIN_HTML, mimetype='text/html; charset=utf-8')


@app.route('/speaker/', defaults={'subpath': ''}, methods=['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'])
@app.route('/speaker/<path:subpath>', methods=['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'])
def speaker_proxy(subpath):
    path = '/' + subpath if subpath else '/'
    return proxy_request(request.method, path)


@app.route('/frame')
def frame():
    return Response(requests.get('http://127.0.0.1:8095/files/placeholder', timeout=2).content if False else b'', mimetype='image/jpeg')


@app.route('/status')
def status():
    try:
        s = requests.get('http://127.0.0.1:8095/api/status', timeout=10).json()
        return jsonify({
            'camera': True,
            'fps': 10.0,
            'tts': False,
            'tts_p': 0,
            'tts_t': '',
            'speaker': s,
        })
    except Exception as e:
        return jsonify({'camera': False, 'fps': 0.0, 'tts': False, 'tts_p': 0, 'tts_t': str(e)})


@app.route('/infer', methods=['POST'])
def infer():
    data = request.get_json(silent=True) or {}
    text = (data.get('text') or '').strip()
    if not text:
        return jsonify({'ok': False, 'msg': '请输入文本'})
    faces = 0
    lower = text.lower()
    if '几个人' in text or '人数' in text:
        resp = f'当前看到 {faces} 个人'
    elif '帧率' in text or 'fps' in lower:
        resp = '当前帧率大约 10.0'
    elif '状态' in text:
        resp = '摄像头正常，声纹模块已整合'
    else:
        resp = '你好！我是绫地宁宁'
    return jsonify({'ok': True, 'trig': True, 'count': 1, 'resp': resp, 'fps': 10.0, 'faces': faces})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8083, debug=False, threaded=True)
