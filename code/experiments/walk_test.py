#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, PoseStamped
import csv
import os


class Go2WalkForward(Node):
    def __init__(self):
        super().__init__('go2_walk_forward')

        # ----------------------------
        # Publishers / Subscribers
        # ----------------------------

        self.wp_pub = self.create_publisher(
            PointStamped,
            '/way_point',
            10
        )

        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/utlidar/robot_pose',
            self.pose_callback,
            10
        )

        # ----------------------------
        # CSV setup
        # ----------------------------

        self.csv_path = os.path.expanduser("~/go2_position_log.csv")
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        # CSV header
        self.csv_writer.writerow([
            "stamp_sec",
            "stamp_nanosec",
            "x",
            "y",
            "z"
        ])

        self.get_logger().info(f"Logging positions to {self.csv_path}")

        # ----------------------------
        # Publish waypoint
        # ----------------------------

        forward_distance = 2.5  # meters

        msg = PointStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x = forward_distance
        msg.point.y = 0.0
        msg.point.z = 0.0

        self.get_logger().info(
            f'Publishing waypoint: x={msg.point.x}, y={msg.point.y}'
        )

        self.wp_pub.publish(msg)

        # Shutdown after some time (adjust if needed)
        self.create_timer(10.0, self.shutdown)

    # ----------------------------
    # Pose Logging
    # ----------------------------

    def pose_callback(self, msg: PoseStamped):
        p = msg.pose.position
        t = msg.header.stamp

        self.csv_writer.writerow([
            t.sec,
            t.nanosec,
            p.x,
            p.y,
            p.z
        ])

    # ----------------------------
    # Shutdown
    # ----------------------------

    def shutdown(self):
        self.get_logger().info("Shutting down, closing CSV.")
        self.csv_file.close()
        rclpy.shutdown()


def main():
    rclpy.init()
    node = Go2WalkForward()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
