#!/usr/bin/env python3
import sys, time, math
import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyUSB0'
BAUD = int(sys.argv[2]) if len(sys.argv) > 2 else 115200

# YDLIDAR triangle/single-channel packet parser
# Packet: PH(2)=0x55AA, CT(1), LSN(1), FSA(2), LSA(2), CS(2), then LSN * 2 bytes samples
# Distance(mm) = Si / 4

HEADER = b'\xaa\x55'


def angle_correct(dist_mm: float) -> float:
    if dist_mm <= 0:
        return 0.0
    return math.degrees(math.atan(21.8 * ((155.3 - dist_mm) / (155.3 * dist_mm))))


def read_exact(ser, n):
    buf = b''
    while len(buf) < n:
        chunk = ser.read(n - len(buf))
        if not chunk:
            break
        buf += chunk
    return buf


def find_header(ser):
    state = 0
    while True:
        b = ser.read(1)
        if not b:
            return False
        if state == 0:
            state = 1 if b == b'\xaa' else 0
        else:
            if b == b'\x55':
                return True
            state = 1 if b == b'\xaa' else 0


def main():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    print(f'open {PORT} @ {BAUD}')
    frames = 0
    try:
        while True:
            if not find_header(ser):
                print('timeout waiting header')
                continue

            head = read_exact(ser, 8)  # CT LSN FSA(2) LSA(2) CS(2)
            if len(head) < 8:
                continue

            ct = head[0]
            lsn = head[1]
            fsa = int.from_bytes(head[2:4], 'little')
            lsa = int.from_bytes(head[4:6], 'little')
            cs = int.from_bytes(head[6:8], 'little')

            if lsn == 0 or lsn > 200:
                continue

            data = read_exact(ser, lsn * 2)
            if len(data) < lsn * 2:
                continue

            # zero packet / sync packet
            if (ct & 0x01) == 0x01:
                sf = (ct >> 1) / 10.0
                print(f'zero packet scan_freq={sf:.1f}Hz')
                continue

            angle_fsa = (fsa >> 1) / 64.0
            angle_lsa = (lsa >> 1) / 64.0
            angle_diff = angle_lsa - angle_fsa
            if angle_diff < 0:
                angle_diff += 360.0

            points = []
            for i in range(lsn):
                si = int.from_bytes(data[i*2:i*2+2], 'little')
                dist = si / 4.0
                if lsn > 1:
                    angle = i * angle_diff / (lsn - 1) + angle_fsa
                else:
                    angle = angle_fsa
                angle += angle_correct(dist)
                if angle >= 360:
                    angle -= 360
                points.append((angle, dist))

            frames += 1
            valid = [(a, d) for a, d in points if d > 1]
            if valid:
                sample = ', '.join([f'({a:6.2f}°, {d:7.1f}mm)' for a, d in valid[:8]])
                print(f'frame {frames:04d} lsn={lsn:3d} start={angle_fsa:6.2f} end={angle_lsa:6.2f} valid={len(valid):3d} :: {sample}')
            else:
                print(f'frame {frames:04d} lsn={lsn:3d} no valid points')

    except KeyboardInterrupt:
        print('\nbye')
    finally:
        ser.close()


if __name__ == '__main__':
    main()
