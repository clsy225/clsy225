#!/usr/bin/env python3
import sys, math, time
import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyUSB0'
BAUD = int(sys.argv[2]) if len(sys.argv) > 2 else 115200
STOP_MM = float(sys.argv[3]) if len(sys.argv) > 3 else 600.0
FRONT_DEG = float(sys.argv[4]) if len(sys.argv) > 4 else 20.0


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


def in_front(angle_deg, front_deg):
    a = angle_deg % 360.0
    return a <= front_deg or a >= (360.0 - front_deg)


def main():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    print(f'min obstacle demo, stop<{STOP_MM}mm, front sector ±{FRONT_DEG}°')
    try:
        while True:
            if not find_header(ser):
                continue
            head = read_exact(ser, 8)
            if len(head) < 8:
                continue
            ct = head[0]
            lsn = head[1]
            fsa = int.from_bytes(head[2:4], 'little')
            lsa = int.from_bytes(head[4:6], 'little')
            if lsn == 0 or lsn > 200:
                continue
            data = read_exact(ser, lsn * 2)
            if len(data) < lsn * 2:
                continue
            if (ct & 0x01) == 0x01:
                continue

            angle_fsa = (fsa >> 1) / 64.0
            angle_lsa = (lsa >> 1) / 64.0
            angle_diff = angle_lsa - angle_fsa
            if angle_diff < 0:
                angle_diff += 360.0

            front = []
            for i in range(lsn):
                si = int.from_bytes(data[i * 2:i * 2 + 2], 'little')
                dist_mm = si / 4.0
                if dist_mm <= 1:
                    continue
                angle_deg = (i * angle_diff / (lsn - 1) + angle_fsa) if lsn > 1 else angle_fsa
                angle_deg += angle_correct(dist_mm)
                if angle_deg >= 360:
                    angle_deg -= 360
                if in_front(angle_deg, FRONT_DEG):
                    front.append((angle_deg, dist_mm))

            if not front:
                continue

            min_angle, min_dist = min(front, key=lambda x: x[1])
            state = 'STOP' if min_dist < STOP_MM else 'GO'
            print(f'{state} nearest_front={min_dist:7.1f}mm angle={min_angle:6.2f}° samples={len(front)}')
            time.sleep(0.05)
    except KeyboardInterrupt:
        print('\nbye')
    finally:
        ser.close()


if __name__ == '__main__':
    main()
