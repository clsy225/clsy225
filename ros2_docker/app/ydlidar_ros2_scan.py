#!/usr/bin/env python3
"""
ROS2 LaserScan publisher for YDLIDAR X3/X3 Pro serial packets.
Requires: rclpy, sensor_msgs
Run after sourcing your ROS2 environment.
"""
import math
import sys
import serial

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import LaserScan
except Exception as e:
    print('ROS2 dependencies missing:', e)
    print('Install/source ROS2 first, then run this node again.')
    raise

PORT = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyUSB0'
BAUD = int(sys.argv[2]) if len(sys.argv) > 2 else 115200
FRAME_ID = sys.argv[3] if len(sys.argv) > 3 else 'laser_frame'


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


class YDLidarNode(Node):
    def __init__(self):
        super().__init__('ydlidar_x3_scan')
        self.pub = self.create_publisher(LaserScan, '/scan', 10)
        self.ser = serial.Serial(PORT, BAUD, timeout=1)
        self.timer = self.create_timer(0.1, self.tick)
        self.ranges = [float('inf')] * 360
        self.scan_freq = 6.0
        self.get_logger().info(f'opened {PORT} @ {BAUD}')

    def tick(self):
        for _ in range(8):
            if not find_header(self.ser):
                return
            head = read_exact(self.ser, 8)
            if len(head) < 8:
                return
            ct = head[0]
            lsn = head[1]
            fsa = int.from_bytes(head[2:4], 'little')
            lsa = int.from_bytes(head[4:6], 'little')
            if lsn == 0 or lsn > 200:
                continue
            data = read_exact(self.ser, lsn * 2)
            if len(data) < lsn * 2:
                return

            if (ct & 0x01) == 0x01:
                sf = (ct >> 1) / 10.0
                if sf > 0:
                    self.scan_freq = sf
                self.publish_scan()
                self.ranges = [float('inf')] * 360
                continue

            angle_fsa = (fsa >> 1) / 64.0
            angle_lsa = (lsa >> 1) / 64.0
            angle_diff = angle_lsa - angle_fsa
            if angle_diff < 0:
                angle_diff += 360.0

            for i in range(lsn):
                si = int.from_bytes(data[i * 2:i * 2 + 2], 'little')
                dist_mm = si / 4.0
                angle_deg = (i * angle_diff / (lsn - 1) + angle_fsa) if lsn > 1 else angle_fsa
                angle_deg += angle_correct(dist_mm)
                angle_deg %= 360.0
                bin_idx = int(round(angle_deg)) % 360
                self.ranges[bin_idx] = float('inf') if dist_mm <= 1 else dist_mm / 1000.0

    def publish_scan(self):
        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = FRAME_ID
        msg.angle_min = 0.0
        msg.angle_max = math.radians(359.0)
        msg.angle_increment = math.radians(1.0)
        msg.time_increment = 0.0
        msg.scan_time = 1.0 / max(self.scan_freq, 0.1)
        msg.range_min = 0.10
        msg.range_max = 8.0
        msg.ranges = self.ranges[:]
        self.pub.publish(msg)

    def destroy_node(self):
        try:
            self.ser.close()
        finally:
            super().destroy_node()


def main():
    rclpy.init()
    node = YDLidarNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
