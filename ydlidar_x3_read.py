#!/usr/bin/env python3
import sys, time, math, json, csv
import serial
from datetime import datetime, timezone

PORT = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyUSB0'
BAUD = int(sys.argv[2]) if len(sys.argv) > 2 else 115200
OUT = sys.argv[3] if len(sys.argv) > 3 else 'console'   # console|jsonl|csv
MAX_FRAMES = int(sys.argv[4]) if len(sys.argv) > 4 else 0  # 0=forever

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


def parse_frame(ser):
    if not find_header(ser):
        return None
    head = read_exact(ser, 8)
    if len(head) < 8:
        return None

    ct = head[0]
    lsn = head[1]
    fsa = int.from_bytes(head[2:4], 'little')
    lsa = int.from_bytes(head[4:6], 'little')
    cs = int.from_bytes(head[6:8], 'little')

    if lsn == 0 or lsn > 200:
        return None

    data = read_exact(ser, lsn * 2)
    if len(data) < lsn * 2:
        return None

    packet = {
        'ct': ct,
        'lsn': lsn,
        'fsa_raw': fsa,
        'lsa_raw': lsa,
        'cs': cs,
        'zero_packet': (ct & 0x01) == 0x01,
    }

    if packet['zero_packet']:
        packet['scan_freq_hz'] = (ct >> 1) / 10.0
        packet['points'] = []
        return packet

    angle_fsa = (fsa >> 1) / 64.0
    angle_lsa = (lsa >> 1) / 64.0
    angle_diff = angle_lsa - angle_fsa
    if angle_diff < 0:
        angle_diff += 360.0

    points = []
    for i in range(lsn):
        si = int.from_bytes(data[i * 2:i * 2 + 2], 'little')
        dist_mm = si / 4.0
        if lsn > 1:
            angle_deg = i * angle_diff / (lsn - 1) + angle_fsa
        else:
            angle_deg = angle_fsa
        angle_deg += angle_correct(dist_mm)
        if angle_deg >= 360:
            angle_deg -= 360
        points.append({
            'index': i,
            'angle_deg': angle_deg,
            'distance_mm': dist_mm,
            'valid': dist_mm > 1.0,
        })

    packet['start_angle_deg'] = angle_fsa
    packet['end_angle_deg'] = angle_lsa
    packet['points'] = points
    return packet


def console_print(frame_idx, pkt):
    if pkt['zero_packet']:
        print(f"zero packet scan_freq={pkt['scan_freq_hz']:.1f}Hz")
        return
    valid = [p for p in pkt['points'] if p['valid']]
    if valid:
        sample = ', '.join([f"({p['angle_deg']:6.2f}°, {p['distance_mm']:7.1f}mm)" for p in valid[:8]])
        print(f"frame {frame_idx:04d} lsn={pkt['lsn']:3d} start={pkt['start_angle_deg']:6.2f} end={pkt['end_angle_deg']:6.2f} valid={len(valid):3d} :: {sample}")
    else:
        print(f"frame {frame_idx:04d} lsn={pkt['lsn']:3d} no valid points")


def main():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    print(f'open {PORT} @ {BAUD}, mode={OUT}')
    frame_idx = 0
    csv_writer = None
    csv_file = None

    if OUT == 'csv':
        csv_file = open('ydlidar_points.csv', 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['timestamp', 'frame', 'point_index', 'angle_deg', 'distance_mm', 'valid'])
    try:
        while True:
            pkt = parse_frame(ser)
            if not pkt:
                continue
            ts = datetime.now(timezone.utc).isoformat()

            if pkt['zero_packet']:
                if OUT == 'console':
                    console_print(frame_idx, pkt)
                elif OUT == 'jsonl':
                    print(json.dumps({'timestamp': ts, 'frame': frame_idx, **pkt}, ensure_ascii=False))
                continue

            frame_idx += 1

            if OUT == 'console':
                console_print(frame_idx, pkt)
            elif OUT == 'jsonl':
                print(json.dumps({'timestamp': ts, 'frame': frame_idx, **pkt}, ensure_ascii=False))
            elif OUT == 'csv':
                for p in pkt['points']:
                    csv_writer.writerow([ts, frame_idx, p['index'], f"{p['angle_deg']:.3f}", f"{p['distance_mm']:.3f}", int(p['valid'])])
                csv_file.flush()

            if MAX_FRAMES and frame_idx >= MAX_FRAMES:
                break
    except KeyboardInterrupt:
        print('\nbye')
    finally:
        ser.close()
        if csv_file:
            csv_file.close()


if __name__ == '__main__':
    main()
