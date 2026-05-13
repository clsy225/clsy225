#!/usr/bin/env python3
import sys, math, time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import serial

PORT = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyUSB0'
BAUD = int(sys.argv[2]) if len(sys.argv) > 2 else 115200


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


def read_points(ser):
    while True:
        if not find_header(ser):
            return []
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

        pts = []
        for i in range(lsn):
            si = int.from_bytes(data[i * 2:i * 2 + 2], 'little')
            dist_mm = si / 4.0
            if dist_mm <= 1:
                continue
            angle_deg = (i * angle_diff / (lsn - 1) + angle_fsa) if lsn > 1 else angle_fsa
            angle_deg += angle_correct(dist_mm)
            if angle_deg >= 360:
                angle_deg -= 360
            r = dist_mm / 1000.0
            theta = math.radians(angle_deg)
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            pts.append((theta, r, x, y))
        return pts


def main():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection='polar')
    ax2 = fig.add_subplot(122)
    scat1 = ax1.scatter([], [], s=8)
    scat2 = ax2.scatter([], [], s=8)

    ax1.set_title('YDLIDAR polar')
    ax1.set_rlim(0, 8)
    ax2.set_title('YDLIDAR XY (meters)')
    ax2.set_xlim(-8, 8)
    ax2.set_ylim(-8, 8)
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True)

    def update(_):
        pts = read_points(ser)
        if not pts:
            return scat1, scat2
        thetas = [p[0] for p in pts]
        rs = [p[1] for p in pts]
        xs = [p[2] for p in pts]
        ys = [p[3] for p in pts]
        scat1.set_offsets(np.c_[thetas, rs])
        scat2.set_offsets(np.c_[xs, ys])
        return scat1, scat2

    ani = FuncAnimation(fig, update, interval=100, cache_frame_data=False)
    plt.tight_layout()
    plt.show()
    ser.close()


if __name__ == '__main__':
    main()
