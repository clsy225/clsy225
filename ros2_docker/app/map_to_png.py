#!/usr/bin/env python3
import os
import sys
import math
from array import array

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from PIL import Image, ImageDraw

OUT = sys.argv[1] if len(sys.argv) > 1 else '/root/ws/map_snapshot.png'
TIMEOUT_SEC = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0

class MapSnap(Node):
    def __init__(self, out_path):
        super().__init__('map_snapshot_saver')
        self.out_path = out_path
        self.got = False
        self.sub = self.create_subscription(OccupancyGrid, '/map', self.cb, 10)
        self.get_logger().info(f'waiting for /map, saving to {out_path}')

    def cb(self, msg: OccupancyGrid):
        w = msg.info.width
        h = msg.info.height
        if w == 0 or h == 0:
            self.get_logger().warning('received empty map')
            return

        img = Image.new('L', (w, h), color=205)
        pixels = img.load()

        data = msg.data
        for y in range(h):
            for x in range(w):
                v = data[y * w + x]
                yy = h - 1 - y
                if v < 0:
                    c = 205
                elif v >= 65:
                    c = 0
                else:
                    c = 255
                pixels[x, yy] = c

        draw = ImageDraw.Draw(img)
        ox = msg.info.origin.position.x
        oy = msg.info.origin.position.y
        res = msg.info.resolution
        if res > 0:
            rx = int(round((-ox) / res))
            ry = int(round((h - 1) - ((-oy) / res)))
            if 0 <= rx < w and 0 <= ry < h:
                draw.line((rx - 5, ry, rx + 5, ry), fill=128, width=1)
                draw.line((rx, ry - 5, rx, ry + 5), fill=128, width=1)

        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        img.save(self.out_path)
        yaml_path = os.path.splitext(self.out_path)[0] + '.txt'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(f'width={w}\n')
            f.write(f'height={h}\n')
            f.write(f'resolution={msg.info.resolution}\n')
            f.write(f'origin_x={msg.info.origin.position.x}\n')
            f.write(f'origin_y={msg.info.origin.position.y}\n')
        self.get_logger().info(f'saved {self.out_path}')
        self.got = True


def main():
    rclpy.init()
    node = MapSnap(OUT)
    end_time = node.get_clock().now().nanoseconds + int(TIMEOUT_SEC * 1e9)
    try:
        while rclpy.ok() and not node.got:
            rclpy.spin_once(node, timeout_sec=0.2)
            if node.get_clock().now().nanoseconds >= end_time:
                node.get_logger().error('timeout waiting for /map')
                break
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(0 if node.got else 1)

if __name__ == '__main__':
    main()
