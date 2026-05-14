#!/usr/bin/env python3
import argparse
import math
import sys
import time

from smbus2 import SMBus

OSC_FREQ = 25_000_000
MODE1 = 0x00
PRESCALE = 0xFE
LED0_ON_L = 0x06
LED0_ON_H = 0x07
LED0_OFF_L = 0x08
LED0_OFF_H = 0x09


def reg_for_channel(ch: int, base: int) -> int:
    return base + 4 * ch


class PCA9685:
    def __init__(self, bus_num: int = 1, address: int = 0x40, freq: float = 50.0):
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
        return prescale

    def set_pwm(self, channel: int, on: int, off: int):
        reg = reg_for_channel(channel, LED0_ON_L)
        self.bus.write_i2c_block_data(self.address, reg, [on & 0xFF, on >> 8, off & 0xFF, off >> 8])

    def set_pulse_us(self, channel: int, pulse_us: float):
        period_us = 1_000_000.0 / self.freq
        counts = int((pulse_us / period_us) * 4096)
        counts = max(0, min(4095, counts))
        self.set_pwm(channel, 0, counts)
        return counts

    def read_channel_regs(self, channel: int):
        base = LED0_ON_L + 4 * channel
        return {
            'ON_L': self.read8(base),
            'ON_H': self.read8(base + 1),
            'OFF_L': self.read8(base + 2),
            'OFF_H': self.read8(base + 3),
        }

    def close(self):
        self.bus.close()


def main():
    p = argparse.ArgumentParser(description='PCA9685 单舵机最小化诊断')
    p.add_argument('--bus', type=int, default=1)
    p.add_argument('--addr', type=lambda x: int(x, 0), default=0x40)
    p.add_argument('--freq', type=float, default=50.0)
    p.add_argument('--channel', type=int, default=0)
    p.add_argument('--dwell', type=float, default=1.5)
    args = p.parse_args()

    pca = PCA9685(bus_num=args.bus, address=args.addr, freq=args.freq)
    try:
        print(f'I2C addr={hex(args.addr)} bus=/dev/i2c-{args.bus}')
        print(f'MODE1=0x{pca.read8(MODE1):02x} PRESCALE=0x{pca.read8(PRESCALE):02x} freq={args.freq}Hz')
        print(f'只测试通道 {args.channel}，其余舵机建议全部拔掉，只保留 1 个舵机 + 独立 5V + 共地')
        sequence = [
            ('neutral', 1500),
            ('min-ish', 1000),
            ('max-ish', 2000),
            ('neutral', 1500),
        ]
        for name, pulse_us in sequence:
            counts = pca.set_pulse_us(args.channel, pulse_us)
            regs = pca.read_channel_regs(args.channel)
            print(f'{name:8s} pulse={pulse_us:4d}us counts={counts:4d} regs={regs}')
            time.sleep(args.dwell)
        print('done')
        return 0
    finally:
        pca.close()


if __name__ == '__main__':
    sys.exit(main())
