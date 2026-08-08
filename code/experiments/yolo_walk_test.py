#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, PoseStamped
import csv
import os
import requests
import cv2
import numpy as np
import pyrealsense2 as rs  # Hardware-specific library
from ultralytics import YOLO

# Preload fix for Jetson memory blocks
os.environ['LD_PRELOAD'] = "/usr/lib/aarch64-linux-gnu/libgomp.so.1"

class Go2WalkForward(Node):
    def __init__(self):
        super().__init__('go2_walk_forward')

        # ----------------------------
        # 1. RealSense Setup
        # ----------------------------
        self.yolo_model = YOLO('yolov8n.pt')
        
        # Initialize RealSense
        self.pipeline = rs.pipeline()
        config = rs.config()
        
        # USE SAFE-MODE SETTINGS (6 FPS) due to the USB 2.0 Hub connection
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 6)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 6)
        
        try:
            self.pipeline.start(config)
            self.get_logger().info("RealSense Depth Camera Started (USB 2.0 Safe Mode)")
        except Exception as e:
            self.get_logger().error(f"Failed to start RealSense: {e}")

        # --- Video Saving Setup ---
        self.video_path = "/home/unitree/GenAssist_Riana/in_line_test/go2_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Standard RealSense frame size at 6 FPS
        self.video_writer = cv2.VideoWriter(self.video_path, fourcc, 6.0, (640, 480))
        # -------------------------------

        self.wp_pub = self.create_publisher(PointStamped, '/way_point', 10)
        self.pose_sub = self.create_subscription(PoseStamped, '/utlidar/robot_pose', self.pose_callback, 10)

        self.csv_path = "/home/unitree/GenAssist_Riana/in_line_test/go2_position_log.csv"
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Added "min_depth_m" column for your research
        self.csv_writer.writerow(["stamp_sec", "stamp_nanosec", "x", "y", "z", "yolo_detections", "vlm_response", "min_depth_m"])

        self.get_logger().info(f"Logging to {self.csv_path}")

        # 1.0m Forward Waypoint
        forward_distance = 1.0
        msg = PointStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x = forward_distance
        self.wp_pub.publish(msg)

        # 10 second mission timer
        self.create_timer(10.0, self.shutdown)

    def pose_callback(self, msg: PoseStamped):
        p = msg.pose.position
        t = msg.header.stamp

        # ----------------------------
        # 2. Get RealSense Data
        # ----------------------------
        try:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                return

            # Convert to numpy arrays
            frame = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # 3. Process Depth (Get distance to center obstacle)
            # Looking at a small ROI in the center of the frame
            roi = depth_image[200:280, 300:340]
            valid_depths = roi[roi > 0]
            min_depth_m = np.min(valid_depths) * 0.001 if valid_depths.size > 0 else 0.0

            # 4. Run YOLO
            results = self.yolo_model(frame, verbose=False)
            detected = [results[0].names[int(c)] for c in results[0].boxes.cls]
            yolo_results_str = ", ".join(detected) if detected else "None"

            # Live Feed & Video
            annotated_frame = results[0].plot()
            cv2.putText(annotated_frame, f"Obstacle: {round(min_depth_m, 2)}m", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Go2 Live RealSense Feed", annotated_frame)
            self.video_writer.write(annotated_frame)
            cv2.waitKey(1)

            # Log data
            self.csv_writer.writerow([t.sec, t.nanosec, p.x, p.y, p.z, yolo_results_str, "N/A", round(min_depth_m, 2)])
            self.csv_file.flush()
        
        except Exception as e:
            self.get_logger().error(f"Error in pose_callback: {e}")

    def shutdown(self):
        self.get_logger().info("Mission Complete. Closing files.")
        
        # 1. Stop the camera pipeline FIRST
        if hasattr(self, 'pipeline'):
            self.pipeline.stop()
        
        # 2. Close UI and Files
        cv2.destroyAllWindows()
        if hasattr(self, 'video_writer'):
            self.video_writer.release()
        self.csv_file.close()
        
        # 3. Force exit the ROS node
        self.destroy_node()
        rclpy.shutdown()
        # Clean exit for Jetson environment
        os._exit(0)

def main():
    rclpy.init()
    node = Go2WalkForward()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.shutdown()

if __name__ == '__main__':
    main()