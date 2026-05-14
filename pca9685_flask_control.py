#!/usr/bin/env python3
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from flask import Flask, jsonify, request, Response

try:
    from smbus2 import SMBus
except Exception:
    SMBus = None

APP = Flask(__name__)
I2C_BUS = 1
I2C_ADDR = 0x40
PWM_FREQ = 50.0
OSC_FREQ = 25_000_000
MODE1 = 0x00
PRESCALE = 0xFE
LED0_ON_L = 0x06
DEFAULT_CHANNELS = [0, 1, 2, 3, 4]
CHANNEL_LIMITS = {
    0: {'min': 10, 'max': 170},
    1: {'min': 55, 'max': 170},
    2: {'min': 10, 'max': 170},
    3: {'min': 10, 'max': 170},
    4: {'min': 45, 'max': 100},
}
CHANNEL_PULSE_US = {
    0: {'min_us': 900, 'max_us': 2100, 'park_min_us': 1000, 'park_max_us': 2000},
    1: {'min_us': 1000, 'max_us': 1950, 'park_min_us': 1050, 'park_max_us': 1900},
    2: {'min_us': 950, 'max_us': 2050, 'park_min_us': 1000, 'park_max_us': 1950},
    3: {'min_us': 950, 'max_us': 2050, 'park_min_us': 1000, 'park_max_us': 1950},
    4: {'min_us': 1100, 'max_us': 1900, 'park_min_us': 1150, 'park_max_us': 1850},
}
DEFAULT_MIN_US = 500
DEFAULT_MAX_US = 2500
DEFAULT_PARK_MIN_US = 1000
DEFAULT_PARK_MAX_US = 2000
PORT = 5050
PAGE_VERSION = '2026-05-14-1700-baseline'

HTML = '''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PCA9685 舵机控制</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 24px; background: #111827; color: #f3f4f6; }
    .muted { color: #9ca3af; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; margin-top: 20px; }
    .card { background: #1f2937; border-radius: 16px; padding: 16px; }
    .row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
    button { background: #2563eb; color: white; border: 0; border-radius: 10px; padding: 10px 14px; cursor: pointer; }
    button.secondary { background: #374151; }
    button.warn { background: #b45309; }
    input[type=range] { width: 100%; }
    .status { margin-top: 12px; padding: 12px; border-radius: 10px; background:#0f172a; }
    .ok { color:#34d399; }
    .bad { color:#f87171; }
    code { background:#0b1220; padding:2px 6px; border-radius:6px; }
    .key { display:inline-block; border:1px solid #6b7280; border-bottom-width: 3px; border-radius:8px; padding:2px 8px; margin: 0 2px; }
    .pill { display:inline-block; border-radius:999px; padding:2px 10px; background:#0b1220; color:#cbd5e1; }
  </style>
</head>
<body>
  <h1>PCA9685 舵机控制</h1>
  <div class="muted">先回到能动的基线版本；暂时不做自动释放消抖。键盘快捷键：<span class="key">1-5</span> 选舵机，<span class="key">←</span>/<span class="key">→</span> 微调，<span class="key">↑</span>/<span class="key">↓</span> 大步调，<span class="key">C</span> 回中，<span class="key">R</span> 释放当前。</div>
  <div class="muted" style="margin-top:8px;">推荐用法：先调到目标角度，再点该通道的 <strong>停止抖动</strong>，只释放当前通道；不要日常使用“释放全部 PWM”。</div>

  <div class="card status" id="status">正在连接...</div>

  <div class="row" style="margin-top:16px; gap:12px;">
    <button onclick="centerAll()">全部回中</button>
    <button class="warn" onclick="releaseAll()">诊断：释放全部 PWM</button>
    <button class="secondary" onclick="refreshState()">刷新状态</button>
    <span class="muted">当前选中通道：<strong id="selectedChannel">0</strong></span>
    <span class="pill">版本 <span id="pageVersion"></span></span>
  </div>

  <div class="row" style="margin-top:12px; gap:16px;">
    <label><input type="checkbox" id="softHold" checked> 轻保持消抖</label>
  </div>

  <div class="grid" id="servoGrid"></div>

<script>
const ANGLE_STEP = 5;
const PAGE_VERSION = '2026-05-14-1805-stop-jitter-fix';
const DEADBAND_DEG = 2.0;
let state = null;
let selectedChannel = 0;
let busy = false;

async function api(path, opts) {
  const options = opts || {};
  const res = await fetch(path + (path.includes('?') ? '&' : '?') + '_=' + Date.now(), Object.assign({
    headers: { 'Content-Type': 'application/json' },
    cache: 'no-store'
  }, options));
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || ('HTTP ' + res.status));
  return data;
}

function render() {
  document.getElementById('selectedChannel').textContent = selectedChannel;
  document.getElementById('pageVersion').textContent = PAGE_VERSION;
  const status = document.getElementById('status');
  const grid = document.getElementById('servoGrid');

  if (!state) {
    status.innerHTML = '<span class="bad">未获取到状态</span>';
    return;
  }

  status.innerHTML = '' +
    '<div><strong>硬件状态：</strong> <span class="' + (state.hardware_ok ? 'ok' : 'bad') + '">' + (state.hardware_ok ? '已连接' : '异常') + '</span></div>' +
    '<div class="muted">I2C: <code>/dev/i2c-' + state.bus + '</code> | 地址: <code>' + state.addr + '</code> | 频率: <code>' + state.freq + 'Hz</code></div>' +
    '<div class="' + (state.last_error ? 'bad' : 'muted') + '">' + (state.last_error ? ('最近错误: ' + state.last_error) : '最近错误: 无') + '</div>' +
    '<div class="muted">当前是可动基线版：先确认 5 个通道能稳定响应，再单独做消抖。</div>';

  grid.innerHTML = '';
  state.servos.forEach(function(servo) {
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = '' +
      '<div class="row" style="justify-content:space-between; margin-bottom:10px;">' +
        '<strong>通道 ' + servo.channel + '</strong>' +
        '<span class="' + (selectedChannel === servo.channel ? 'ok' : 'muted') + '">' + (selectedChannel === servo.channel ? '当前选中' : '') + '</span>' +
      '</div>' +
      '<div class="muted">当前角度：<span id="angleLabel_' + servo.channel + '">' + Number(servo.angle).toFixed(1) + '</span>°</div>' +
      '<div class="muted">范围：' + servo.min_angle + '° ~ ' + servo.max_angle + '°</div>' +
      '<input type="range" min="' + servo.min_angle + '" max="' + servo.max_angle + '" step="1" value="' + servo.angle + '" oninput="previewAngle(' + servo.channel + ', this.value)" onchange="commitAngle(' + servo.channel + ', this.value)">' +
      '<div class="row" style="margin-top: 12px;">' +
        '<button class="secondary" onclick="nudge(' + servo.channel + ', -5)">-5°</button>' +
        '<button class="secondary" onclick="nudge(' + servo.channel + ', 5)">+5°</button>' +
        '<button onclick="setAngle(' + servo.channel + ', 90)">回中</button>' +
        '<button class="warn" onclick="stopJitter(' + servo.channel + ')">停止抖动</button>' +
        '<button class="secondary" onclick="selectChannel(' + servo.channel + ')">选中</button>' +
      '</div>';
    grid.appendChild(card);
  });
}

async function refreshState() {
  try {
    state = await api('/api/state');
    if (!state.servos.find(function(s) { return s.channel === selectedChannel; }) && state.servos.length) {
      selectedChannel = state.servos[0].channel;
    }
    render();
  } catch (e) {
    document.getElementById('status').innerHTML = '<span class="bad">' + e.message + '</span>';
  }
}

function selectChannel(channel) {
  selectedChannel = channel;
  render();
}

function previewAngle(channel, angle) {
  const label = document.getElementById('angleLabel_' + channel);
  if (label) label.textContent = Number(angle).toFixed(1);
}

async function setAngle(channel, angle) {
  if (busy) return;
  const next = Number(angle);
  const servo = state && state.servos ? state.servos.find(function(s) { return s.channel === channel; }) : null;
  if (servo) {
    const current = Number(servo.angle);
    if (Math.abs(next - current) < DEADBAND_DEG) {
      return;
    }
  }
  busy = true;
  try {
    state = await api('/api/set_angle', {
      method: 'POST',
      body: JSON.stringify({ channel: channel, angle: next })
    });
    render();
  } catch (e) {
    alert(e.message);
  } finally {
    busy = false;
  }
}

async function commitAngle(channel, angle) {
  await setAngle(channel, angle);
}

async function nudge(channel, delta) {
  if (!state) return;
  const servo = state.servos.find(function(s) { return s.channel === channel; });
  if (!servo) return;
  await setAngle(channel, Number(servo.angle) + delta);
}

async function stopJitter(channel) {
  if (busy) return;
  busy = true;
  try {
    const useSoftHold = document.getElementById('softHold') && document.getElementById('softHold').checked;
    const path = useSoftHold ? '/api/park' : '/api/release';
    state = await api(path, {
      method: 'POST',
      body: JSON.stringify({ channel: channel })
    });
    render();
  } catch (e) {
    alert(e.message);
  } finally {
    busy = false;
  }
}

async function releaseOne(channel) {
  try {
    state = await api('/api/release', {
      method: 'POST',
      body: JSON.stringify({ channel: channel })
    });
    render();
  } catch (e) {
    alert(e.message);
  }
}

async function releaseAll() {
  const ok = window.confirm('这是诊断按钮：会让 5 个舵机全部停止保持位置。正常消抖请用单个通道的“停止抖动”。确定继续吗？');
  if (!ok) return;
  try {
    state = await api('/api/release_all', { method: 'POST' });
    render();
  } catch (e) {
    alert(e.message);
  }
}

async function centerAll() {
  try {
    state = await api('/api/center_all', { method: 'POST' });
    render();
  } catch (e) {
    alert(e.message);
  }
}

document.addEventListener('keydown', async function(e) {
  if (!state) return;
  const tag = document.activeElement ? document.activeElement.tagName : '';
  if (tag === 'INPUT' || tag === 'TEXTAREA') return;

  if (e.key >= '1' && e.key <= '5') {
    const idx = Number(e.key) - 1;
    if (state.servos[idx]) {
      selectedChannel = state.servos[idx].channel;
      render();
    }
    return;
  }

  const servo = state.servos.find(function(s) { return s.channel === selectedChannel; });
  if (!servo) return;

  if (e.key === 'ArrowLeft') { e.preventDefault(); await setAngle(selectedChannel, Number(servo.angle) - ANGLE_STEP); }
  else if (e.key === 'ArrowRight') { e.preventDefault(); await setAngle(selectedChannel, Number(servo.angle) + ANGLE_STEP); }
  else if (e.key === 'ArrowUp') { e.preventDefault(); await setAngle(selectedChannel, Number(servo.angle) + 3 * ANGLE_STEP); }
  else if (e.key === 'ArrowDown') { e.preventDefault(); await setAngle(selectedChannel, Number(servo.angle) - 3 * ANGLE_STEP); }
  else if (e.key.toLowerCase() === 'c') { e.preventDefault(); await setAngle(selectedChannel, 90); }
  else if (e.key.toLowerCase() === 'r') { e.preventDefault(); await releaseOne(selectedChannel); }
});

window.addEventListener('load', function() {
  refreshState();
});
</script>
</body>
</html>
'''


def channel_limits(channel: int):
    limits = CHANNEL_LIMITS.get(channel, {'min': 0, 'max': 180})
    lo = float(limits['min'])
    hi = float(limits['max'])
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def channel_pulse(channel: int, park: bool = False):
    cfg = CHANNEL_PULSE_US.get(channel, {})
    if park:
        return (
            float(cfg.get('park_min_us', DEFAULT_PARK_MIN_US)),
            float(cfg.get('park_max_us', DEFAULT_PARK_MAX_US)),
        )
    return (
        float(cfg.get('min_us', DEFAULT_MIN_US)),
        float(cfg.get('max_us', DEFAULT_MAX_US)),
    )


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class PCA9685:
    def __init__(self, bus_num: int = I2C_BUS, address: int = I2C_ADDR, freq: float = PWM_FREQ):
        if SMBus is None:
            raise RuntimeError('缺少 smbus2，请先安装')
        self.address = address
        self.freq = freq
        self.bus = SMBus(bus_num)
        self.write8(MODE1, 0x00)
        self.set_pwm_freq(freq)

    def write8(self, reg: int, value: int):
        self.bus.write_byte_data(self.address, reg, value & 0xFF)

    def read8(self, reg: int) -> int:
        return self.bus.read_byte_data(self.address, reg)

    def set_pwm_freq(self, freq_hz: float):
        prescaleval = OSC_FREQ / 4096.0 / float(freq_hz) - 1.0
        prescale = max(3, min(255, int(math.floor(prescaleval + 0.5))))
        oldmode = self.read8(MODE1)
        self.write8(MODE1, (oldmode & 0x7F) | 0x10)
        self.write8(PRESCALE, prescale)
        self.write8(MODE1, oldmode)
        time.sleep(0.005)
        self.write8(MODE1, oldmode | 0xA1)

    def set_pwm(self, channel: int, on: int, off: int):
        reg = LED0_ON_L + 4 * channel
        self.bus.write_i2c_block_data(self.address, reg, [on & 0xFF, on >> 8, off & 0xFF, off >> 8])

    def set_angle(self, channel: int, angle: float):
        lo, hi = channel_limits(channel)
        angle = clamp(angle, lo, hi)
        min_us, max_us = channel_pulse(channel, park=False)
        pulse_us = min_us + (max_us - min_us) * (angle / 180.0)
        period_us = 1_000_000.0 / self.freq
        off = int((pulse_us / period_us) * 4096)
        self.set_pwm(channel, 0, max(0, min(4095, off)))

    def park_angle(self, channel: int, angle: float):
        lo, hi = channel_limits(channel)
        angle = clamp(angle, lo, hi)
        min_us, max_us = channel_pulse(channel, park=True)
        pulse_us = min_us + (max_us - min_us) * (angle / 180.0)
        period_us = 1_000_000.0 / self.freq
        off = int((pulse_us / period_us) * 4096)
        self.set_pwm(channel, 0, max(0, min(4095, off)))

    def release(self, channel: int):
        self.set_pwm(channel, 0, 0)


@dataclass
class ServoState:
    channel: int
    angle: float = 90.0


class ServoController:
    def __init__(self, channels):
        self.channels = channels
        self.servos = {ch: ServoState(channel=ch) for ch in channels}
        self.lock = threading.Lock()
        self.last_error = ''
        self.hardware_ok = False
        self.driver = None
        self._connect()

    def _connect(self):
        try:
            self.driver = PCA9685()
            for ch in self.channels:
                self.driver.set_angle(ch, self.servos[ch].angle)
            self.hardware_ok = True
            self.last_error = ''
        except Exception as e:
            self.driver = None
            self.hardware_ok = False
            self.last_error = str(e)

    def _ensure(self):
        if self.driver is None:
            self._connect()
        if self.driver is None:
            raise RuntimeError(self.last_error or 'PCA9685 未连接')

    def set_angle(self, channel, angle):
        with self.lock:
            self._ensure()
            lo, hi = channel_limits(channel)
            angle = clamp(float(angle), lo, hi)
            self.driver.set_angle(channel, angle)
            self.servos[channel].angle = angle
            self.hardware_ok = True
            self.last_error = ''

    def center_all(self):
        with self.lock:
            self._ensure()
            errors = []
            for ch in self.channels:
                try:
                    self.driver.set_angle(ch, 90.0)
                    self.servos[ch].angle = 90.0
                except Exception as e:
                    errors.append(f'ch{ch}: {e}')
            self.hardware_ok = len(errors) == 0
            self.last_error = '; '.join(errors)
            if len(errors) == len(self.channels):
                raise RuntimeError(self.last_error or '全部回中失败')

    def release(self, channel):
        with self.lock:
            self._ensure()
            self.driver.release(channel)
            self.hardware_ok = True
            self.last_error = ''

    def park(self, channel):
        with self.lock:
            self._ensure()
            angle = self.servos[channel].angle
            self.driver.park_angle(channel, angle)
            self.hardware_ok = True
            self.last_error = ''

    def release_all(self):
        with self.lock:
            self._ensure()
            for ch in self.channels:
                self.driver.release(ch)
            self.hardware_ok = True
            self.last_error = ''

    def state(self):
        return {
            'bus': I2C_BUS,
            'addr': hex(I2C_ADDR),
            'freq': PWM_FREQ,
            'hardware_ok': self.hardware_ok,
            'last_error': self.last_error,
            'servos': [
                {
                    **self.servos[ch].__dict__,
                    'min_angle': channel_limits(ch)[0],
                    'max_angle': channel_limits(ch)[1],
                    'min_us': channel_pulse(ch, park=False)[0],
                    'max_us': channel_pulse(ch, park=False)[1],
                }
                for ch in self.channels
            ],
        }


controller = ServoController(DEFAULT_CHANNELS)


@APP.get('/')
def index():
    return Response(HTML, mimetype='text/html', headers={'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0'})


@APP.get('/api/state')
def api_state():
    return jsonify(controller.state())


@APP.post('/api/set_angle')
def api_set_angle():
    data = request.get_json(force=True)
    controller.set_angle(int(data['channel']), float(data['angle']))
    return jsonify(controller.state())


@APP.post('/api/release')
def api_release():
    data = request.get_json(force=True)
    controller.release(int(data['channel']))
    return jsonify(controller.state())


@APP.post('/api/park')
def api_park():
    data = request.get_json(force=True)
    controller.park(int(data['channel']))
    return jsonify(controller.state())


@APP.post('/api/release_all')
def api_release_all():
    controller.release_all()
    return jsonify(controller.state())


@APP.post('/api/center_all')
def api_center_all():
    controller.center_all()
    return jsonify(controller.state())


@APP.errorhandler(Exception)
def handle_error(e):
    controller.hardware_ok = False
    controller.last_error = str(e)
    return jsonify({'error': str(e), 'state': controller.state()}), 500


if __name__ == '__main__':
    print(f'Open http://<树莓派IP>:{PORT} in your browser')
    APP.run(host='0.0.0.0', port=PORT, debug=False)
