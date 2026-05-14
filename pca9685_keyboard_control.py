#!/usr/bin/env python3
import argparse
import math
import sys
import time
from datetime import datetime

try:
    from smbus2 import SMBus
except Exception:
    SMBus = None

OSC_FREQ = 25_000_000
MODE1 = 0x00
PRESCALE = 0xFE
LED0_ON_L = 0x06


class PCA9685:
    def __init__(self, bus_num: int = 1, address: int = 0x40, freq: float = 50.0):
        if SMBus is None:
            raise RuntimeError('缺少 smbus2，请先安装 smbus2 / python3-smbus')
        self.bus = SMBus(bus_num)
        self.address = address
        self.freq = freq
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
        sleep = (oldmode & 0x7F) | 0x10
        self.write8(MODE1, sleep)
        self.write8(PRESCALE, prescale)
        self.write8(MODE1, oldmode)
        time.sleep(0.005)
        self.write8(MODE1, oldmode | 0xA1)

    def set_pwm(self, channel: int, on: int, off: int):
        reg = LED0_ON_L + 4 * channel
        self.bus.write_i2c_block_data(self.address, reg, [on & 0xFF, on >> 8, off & 0xFF, off >> 8])

    def angle_to_counts(self, angle: float, min_us: int = 500, max_us: int = 2500):
        angle = max(0.0, min(180.0, angle))
        pulse = min_us + (max_us - min_us) * (angle / 180.0)
        period_us = 1_000_000.0 / self.freq
        counts = int((pulse / period_us) * 4096)
        return max(0, min(4095, counts)), pulse

    def set_angle(self, channel: int, angle: float, min_us: int = 500, max_us: int = 2500):
        counts, pulse = self.angle_to_counts(angle, min_us=min_us, max_us=max_us)
        self.set_pwm(channel, 0, counts)
        return counts, pulse

    def release(self, channel: int):
        self.set_pwm(channel, 0, 0)

    def close(self):
        self.bus.close()


def parse_args():
    p = argparse.ArgumentParser(description='PCA9685 键盘输入角度控制（5舵机现场调试版）')
    p.add_argument('--bus', type=int, default=1)
    p.add_argument('--addr', type=lambda x: int(x, 0), default=0x40)
    p.add_argument('--freq', type=float, default=50.0)
    p.add_argument('--min-us', type=int, default=500)
    p.add_argument('--max-us', type=int, default=2500)
    p.add_argument('--default-angle', type=float, default=90.0)
    p.add_argument('--channels', default='0,1,2,3,4', help='允许控制的通道列表，例如 0,1,2,3,4')
    p.add_argument('--step', type=float, default=5.0, help='默认步进角度')
    p.add_argument('--big-step', type=float, default=15.0, help='大步进角度')
    p.add_argument('--pause', type=float, default=0.05, help='两次动作间最小停顿')
    return p.parse_args()


def print_help(channels):
    print('\n可用命令：')
    print('  list / status                查看当前全部状态')
    print('  select <通道>                选择当前活动通道，例如: select 2')
    print('  <通道> <角度>                例如: 0 90')
    print('  set <通道> <角度>            例如: set 3 120')
    print('  current <角度>               设置当前活动通道角度')
    print('  all <角度>                   所有通道一起转，例如: all 45')
    print('  step <通道> <增量>           例如: step 2 -10')
    print('  nudge <增量>                 当前活动通道微调，例如: nudge 5')
    print('  left / right                 当前活动通道 ±默认步进')
    print('  up / down                    当前活动通道 ±大步进')
    print('  step-size <角度>             设置默认步进，例如: step-size 3')
    print('  big-step <角度>              设置大步进，例如: big-step 12')
    print('  center                       所有通道回到 90°')
    print('  center <通道>                单个通道回中')
    print('  release <通道>               释放单个通道 PWM')
    print('  release all                  释放全部通道 PWM')
    print('  diag                         打印 I2C / PWM 关键状态')
    print('  help                         显示帮助')
    print('  quit / exit                  退出程序')
    print(f'\n当前允许通道: {channels}\n')


def clamp(angle: float) -> float:
    return max(0.0, min(180.0, angle))


def now_ts() -> str:
    return datetime.now().strftime('%H:%M:%S')


def print_status(channels, current, selected, step_size, big_step, pca, min_us, max_us):
    print('\n=== 当前状态 ===')
    print(f'时间: {now_ts()} | 当前通道: {selected} | 默认步进: {step_size:.1f}° | 大步进: {big_step:.1f}°')
    print(f'I2C: addr={hex(pca.address)} freq={pca.freq}Hz MODE1=0x{pca.read8(MODE1):02x} PRESCALE=0x{pca.read8(PRESCALE):02x}')
    for ch in channels:
        counts, pulse = pca.angle_to_counts(current[ch], min_us=min_us, max_us=max_us)
        mark = ' <==' if ch == selected else ''
        print(f'  通道 {ch}: {current[ch]:6.1f}° | pulse={pulse:7.1f}us | counts={counts:4d}{mark}')
    print('================\n')


def apply_angle(pca, current, channel, angle, min_us, max_us, pause_s):
    angle = clamp(angle)
    counts, pulse = pca.set_angle(channel, angle, min_us=min_us, max_us=max_us)
    current[channel] = angle
    print(f'[{now_ts()}] 通道 {channel} -> {angle:5.1f}° | pulse={pulse:7.1f}us | counts={counts:4d}')
    if pause_s > 0:
        time.sleep(pause_s)


def main():
    args = parse_args()
    channels = [int(x.strip()) for x in args.channels.split(',') if x.strip()]
    if not channels:
        print('没有有效通道')
        return 2

    current = {ch: clamp(args.default_angle) for ch in channels}
    selected = channels[0]
    step_size = float(args.step)
    big_step = float(args.big_step)
    pca = PCA9685(bus_num=args.bus, address=args.addr, freq=args.freq)

    try:
        for ch, angle in current.items():
            pca.set_angle(ch, angle, min_us=args.min_us, max_us=args.max_us)

        print('PCA9685 键盘控制已启动（5舵机现场调试版）')
        print(f'I2C 地址: {hex(args.addr)} | 频率: {args.freq}Hz | 当前活动通道: {selected}')
        print_help(channels)
        print_status(channels, current, selected, step_size, big_step, pca, args.min_us, args.max_us)

        while True:
            try:
                raw = input(f'servo[ch={selected}]> ').strip()
            except (EOFError, KeyboardInterrupt):
                print('\n退出')
                break

            if not raw:
                continue

            parts = raw.split()
            cmd = parts[0].lower()

            try:
                if cmd in ('quit', 'exit'):
                    break
                elif cmd == 'help':
                    print_help(channels)
                elif cmd in ('list', 'status', 's'):
                    print_status(channels, current, selected, step_size, big_step, pca, args.min_us, args.max_us)
                elif cmd == 'diag':
                    print(f'[{now_ts()}] DIAG addr={hex(pca.address)} MODE1=0x{pca.read8(MODE1):02x} PRESCALE=0x{pca.read8(PRESCALE):02x}')
                elif cmd == 'select' and len(parts) == 2:
                    ch = int(parts[1])
                    if ch not in current:
                        print('无效通道')
                        continue
                    selected = ch
                    print(f'[{now_ts()}] 当前活动通道 -> {selected}')
                elif cmd == 'center' and len(parts) == 1:
                    for ch in channels:
                        apply_angle(pca, current, ch, 90.0, args.min_us, args.max_us, args.pause)
                    print(f'[{now_ts()}] 所有通道已回中到 90°')
                elif cmd == 'center' and len(parts) == 2:
                    ch = int(parts[1])
                    if ch not in current:
                        print('无效通道')
                        continue
                    apply_angle(pca, current, ch, 90.0, args.min_us, args.max_us, args.pause)
                elif cmd == 'all' and len(parts) == 2:
                    angle = clamp(float(parts[1]))
                    for ch in channels:
                        apply_angle(pca, current, ch, angle, args.min_us, args.max_us, args.pause)
                    print(f'[{now_ts()}] 所有通道 -> {angle:.1f}°')
                elif cmd == 'step' and len(parts) == 3:
                    ch = int(parts[1])
                    delta = float(parts[2])
                    if ch not in current:
                        print('无效通道')
                        continue
                    apply_angle(pca, current, ch, current[ch] + delta, args.min_us, args.max_us, args.pause)
                elif cmd == 'nudge' and len(parts) == 2:
                    delta = float(parts[1])
                    apply_angle(pca, current, selected, current[selected] + delta, args.min_us, args.max_us, args.pause)
                elif cmd == 'left':
                    apply_angle(pca, current, selected, current[selected] - step_size, args.min_us, args.max_us, args.pause)
                elif cmd == 'right':
                    apply_angle(pca, current, selected, current[selected] + step_size, args.min_us, args.max_us, args.pause)
                elif cmd == 'up':
                    apply_angle(pca, current, selected, current[selected] + big_step, args.min_us, args.max_us, args.pause)
                elif cmd == 'down':
                    apply_angle(pca, current, selected, current[selected] - big_step, args.min_us, args.max_us, args.pause)
                elif cmd == 'step-size' and len(parts) == 2:
                    step_size = max(0.1, float(parts[1]))
                    print(f'[{now_ts()}] 默认步进 -> {step_size:.1f}°')
                elif cmd == 'big-step' and len(parts) == 2:
                    big_step = max(0.1, float(parts[1]))
                    print(f'[{now_ts()}] 大步进 -> {big_step:.1f}°')
                elif cmd == 'release' and len(parts) == 2:
                    if parts[1].lower() == 'all':
                        for ch in channels:
                            pca.release(ch)
                            print(f'[{now_ts()}] 已释放通道 {ch} PWM')
                            if args.pause > 0:
                                time.sleep(args.pause)
                        print(f'[{now_ts()}] 已释放所有通道 PWM')
                    else:
                        ch = int(parts[1])
                        if ch not in current:
                            print('无效通道')
                            continue
                        pca.release(ch)
                        print(f'[{now_ts()}] 已释放通道 {ch} PWM')
                elif cmd == 'current' and len(parts) == 2:
                    angle = clamp(float(parts[1]))
                    apply_angle(pca, current, selected, angle, args.min_us, args.max_us, args.pause)
                elif cmd == 'set' and len(parts) == 3:
                    ch = int(parts[1])
                    angle = clamp(float(parts[2]))
                    if ch not in current:
                        print('无效通道')
                        continue
                    apply_angle(pca, current, ch, angle, args.min_us, args.max_us, args.pause)
                elif len(parts) == 2:
                    ch = int(parts[0])
                    angle = clamp(float(parts[1]))
                    if ch not in current:
                        print('无效通道')
                        continue
                    apply_angle(pca, current, ch, angle, args.min_us, args.max_us, args.pause)
                else:
                    print('命令格式不对，输入 help 查看用法')
            except ValueError:
                print('参数格式不对，输入 help 查看用法')
            except Exception as e:
                print(f'[{now_ts()}] 操作失败: {e}')
                print('建议先检查: V+ / 共地 / 三线方向 / I2C 是否掉线')
    finally:
        pca.close()

    return 0


if __name__ == '__main__':
    sys.exit(main())
