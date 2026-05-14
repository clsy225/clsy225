#!/usr/bin/env python3
import argparse
import math
import sys
import time

try:
    from smbus2 import SMBus
except Exception:
    SMBus = None

OSC_FREQ = 25_000_000
MODE1 = 0x00
PRESCALE = 0xFE
LED0_ON_L = 0x06


def i2c_probe(addr: int, bus_num: int) -> bool:
    if SMBus is None:
        return True
    try:
        with SMBus(bus_num) as bus:
            bus.write_quick(addr)
        return True
    except Exception:
        return False


class PCA9685:
    def __init__(self, bus_num: int = 1, address: int = 0x40, freq: float = 50.0):
        if SMBus is None:
            raise RuntimeError('缺少 smbus2，请先安装: sudo apt-get install -y python3-smbus i2c-tools 或 pip install smbus2')
        self.bus = SMBus(bus_num)
        self.address = address
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

    def set_pulse_us(self, channel: int, pulse_us: float, freq_hz: float = 50.0):
        period_us = 1_000_000.0 / freq_hz
        counts = int((pulse_us / period_us) * 4096)
        counts = max(0, min(4095, counts))
        self.set_pwm(channel, 0, counts)

    def set_angle(self, channel: int, angle: float, min_us: int = 500, max_us: int = 2500, freq_hz: float = 50.0):
        angle = max(0.0, min(180.0, angle))
        pulse = min_us + (max_us - min_us) * (angle / 180.0)
        self.set_pulse_us(channel, pulse, freq_hz)

    def release(self, channel: int):
        self.set_pwm(channel, 0, 0)

    def close(self):
        self.bus.close()


def main():
    parser = argparse.ArgumentParser(description='PCA9685 5路舵机测试')
    parser.add_argument('--bus', type=int, default=1)
    parser.add_argument('--addr', type=lambda x: int(x, 0), default=0x40)
    parser.add_argument('--freq', type=float, default=50.0)
    parser.add_argument('--channels', default='0,1,2,3,4', help='逗号分隔，例如 0,1,2,3,4')
    parser.add_argument('--min-us', type=int, default=500)
    parser.add_argument('--max-us', type=int, default=2500)
    parser.add_argument('--dwell', type=float, default=0.8, help='每个位置停留秒数')
    parser.add_argument('--release', action='store_true', help='结束后释放PWM输出')
    args = parser.parse_args()

    channels = [int(x.strip()) for x in args.channels.split(',') if x.strip()]
    if not channels:
        print('没有有效通道')
        return 2

    if not i2c_probe(args.addr, args.bus):
        print(f'未在 /dev/i2c-{args.bus} 上探测到 PCA9685 地址 {hex(args.addr)}')
        print('请确认: VCC/GND, SDA(GPIO2 pin3), SCL(GPIO3 pin5), 模块地址焊点(A0~A5)')
        return 3

    pca = PCA9685(bus_num=args.bus, address=args.addr, freq=args.freq)
    try:
        positions = [90, 30, 150, 90]
        for ch in channels:
            print(f'测试通道 {ch}...')
            for angle in positions:
                print(f'  -> {angle}°')
                pca.set_angle(ch, angle, min_us=args.min_us, max_us=args.max_us, freq_hz=args.freq)
                time.sleep(args.dwell)
        print('测试完成')
        if args.release:
            for ch in channels:
                pca.release(ch)
    finally:
        pca.close()


if __name__ == '__main__':
    sys.exit(main())
