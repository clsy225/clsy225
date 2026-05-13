#!/usr/bin/env python3
import sys
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

try:
    from cv_bridge import CvBridge
    import cv2
except Exception as e:
    print('missing dependency:', e)
    raise

OUT = sys.argv[1] if len(sys.argv) > 1 else '/root/ws/camera_snapshot.png'
TIMEOUT_SEC = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
TOPIC = sys.argv[3] if len(sys.argv) > 3 else '/image_raw'

class CameraSnap(Node):
    def __init__(self, out_path, topic):
        super().__init__('camera_snapshot_saver')
        self.out_path = out_path
        self.bridge = CvBridge()
        self.got = False
        self.sub = self.create_subscription(Image, topic, self.cb, 10)
        self.get_logger().info(f'waiting for {topic}, saving to {out_path}')

    def cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
            cv2.imwrite(self.out_path, frame)
            self.get_logger().info(f'saved {self.out_path}')
            self.got = True
        except Exception as e:
            self.get_logger().error(f'failed to save image: {e}')


def main():
    rclpy.init()
    node = CameraSnap(OUT, TOPIC)
    end_time = node.get_clock().now().nanoseconds + int(TIMEOUT_SEC * 1e9)
    try:
        while rclpy.ok() and not node.got:
            rclpy.spin_once(node, timeout_sec=0.2)
            if node.get_clock().now().nanoseconds >= end_time:
                node.get_logger().error(f'timeout waiting for {TOPIC}')
                break
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(0 if node.got else 1)

if __name__ == '__main__':
    main()
