#!/usr/bin/env python3
import time
from smbus2 import SMBus

ADDR = 0x40
BUS_NUM = 1
MODE1 = 0x00
PRESCALE = 0xFE
LED0_ON_L = 0x06
ALLLED_ON_L = 0xFA
ALLLED_OFF_L = 0xFC

SERVOMIN = 150  # Adafruit classic example value
SERVOMAX = 600  # Adafruit classic example value


def write8(bus, reg, val):
    bus.write_byte_data(ADDR, reg, val & 0xFF)


def read8(bus, reg):
    return bus.read_byte_data(ADDR, reg)


def set_pwm(bus, ch, on, off):
    reg = LED0_ON_L + 4 * ch
    bus.write_i2c_block_data(ADDR, reg, [on & 0xFF, (on >> 8) & 0xFF, off & 0xFF, (off >> 8) & 0xFF])


def set_all_pwm(bus, on, off):
    bus.write_i2c_block_data(ADDR, ALLLED_ON_L, [on & 0xFF, (on >> 8) & 0xFF, off & 0xFF, (off >> 8) & 0xFF])


def reset(bus):
    write8(bus, MODE1, 0x00)


def set_pwm_freq(bus, freq):
    freq *= 0.9  # match old Adafruit library behavior
    prescaleval = 25000000.0
    prescaleval /= 4096.0
    prescaleval /= float(freq)
    prescaleval -= 1.0
    prescale = int(prescaleval + 0.5)
    oldmode = read8(bus, MODE1)
    newmode = (oldmode & 0x7F) | 0x10
    write8(bus, MODE1, newmode)
    write8(bus, PRESCALE, prescale)
    write8(bus, MODE1, oldmode)
    time.sleep(0.005)
    write8(bus, MODE1, oldmode | 0xA1)
    return prescale


def main():
    with SMBus(BUS_NUM) as bus:
        print('Reset PCA9685...')
        reset(bus)
        time.sleep(0.01)
        prescale = set_pwm_freq(bus, 60)
        print(f'MODE1=0x{read8(bus, MODE1):02x}, PRESCALE=0x{read8(bus, PRESCALE):02x} (calc {prescale})')

        print('Set all channels to center-like pulse...')
        for ch in range(16):
            set_pwm(bus, ch, 0, 375)
        time.sleep(1.0)

        print('Sweep all 16 channels together: SERVOMIN -> SERVOMAX -> middle')
        for off in (SERVOMIN, 375, SERVOMAX, 375):
            print(f'  off={off}')
            for ch in range(16):
                set_pwm(bus, ch, 0, off)
            time.sleep(1.5)

        print('Now test channels 0..4 individually with wide sweep')
        for ch in range(5):
            print(f'  channel {ch}')
            for off in (SERVOMIN, 375, SERVOMAX, 375):
                set_pwm(bus, ch, 0, off)
                time.sleep(0.8)

        print('Release all outputs')
        for ch in range(16):
            set_pwm(bus, ch, 0, 0)


if __name__ == '__main__':
    main()
