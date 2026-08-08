#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, PoseStamped
import csv
import os
import base64
import requests
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import threading
import time

# Fix for OpenMP conflict on some ARM devices (Jetson/Orin)
os.environ['LD_PRELOAD'] = "/usr/lib/aarch64-linux-gnu/libgomp.so.1"

class Go2WalkForward(Node):
    def __init__(self):
        super().__init__('go2_walk_forward')

        # ----------------------------
        # 1. Vision & VLM Setup
        # ----------------------------
        self.yolo_model = YOLO('yolov8n.pt')
        self.vlm_endpoint = "http://localhost:5000/vlm"
        
        self.vlm_busy = False
        self.vlm_description = "Waiting for first VLM trigger"
        self.vlm_inference_duration = None  
        self.latest_frame = None
        
        # RealSense Setup
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        
        try:
            self.pipeline.start(config)
            self.get_logger().info("RealSense Started (15 Stability Mode)")
        except Exception as e:
            self.get_logger().error(f"Hardware Error: {e}")

        self.video_path = "/home/unitree/GenAssist_Riana/in_line_test/go2_output_vlm_granite.mp4"
        self.video_writer = cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*'mp4v'), 6.0, (640, 480))

        # ----------------------------
        # 2. ROS2 Communication
        # ----------------------------
        self.wp_pub = self.create_publisher(PointStamped, '/way_point', 10)
        self.pose_sub = self.create_subscription(PoseStamped, '/utlidar/robot_pose', self.pose_callback, 10)
        
        self.csv_path = "/home/unitree/GenAssist_Riana/in_line_test/go2_position_log_vlm_moondream.csv"
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["stamp_sec", "stamp_nanosec", "x", "y", "z", "yolo_detections", "yolo_confidence", "vlm_response", "vlm_inference_time", "min_depth_m"])

        self.get_logger().info(f"Logging to {self.csv_path}")

        # Timers
        self.create_timer(0.16, self.capture_loop) 
        self.move_timer = self.create_timer(1.0, self.publish_waypoint)
        self.create_timer(40.0, self.shutdown)

    def publish_waypoint(self):
        if self.vlm_busy:
            return

        msg = PointStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x = 2.5 
        self.wp_pub.publish(msg)
        
    def capture_loop(self):
        """This function runs 6 times a second regardless of VLM status."""
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=500)
            color_frame = frames.get_color_frame()
            if color_frame:
                # Capture the fresh raw frame
                new_frame = np.asanyarray(color_frame.get_data())
                
                # If we aren't busy with YOLO/VLM, update the shared frame with raw data
                if not self.vlm_busy or self.latest_frame is None:
                    self.latest_frame = new_frame
                
                # Always record whatever is in latest_frame (annotated or raw)
                self.video_writer.write(self.latest_frame)
        except Exception:
            pass

    def reset_vlm_status(self):
        self.vlm_description = None

    def vlm_request_worker(self, frame):
        start_time = time.time() 
        try:
            # 1. Physical Settle: Let vibrations stop before processing
            time.sleep(0.2) 
            
            # 2. Resize: Lower resolution = faster inference & lower token count
            small_frame = cv2.resize(frame, (320, 240)) 
            _, buffer = cv2.imencode('.jpg', small_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # 3. Payload: Concise prompt to keep evaluation under 500 tokens
            payload = {
                "prompt": "Describe the path ahead and any obstacles while giving navigation instructions to keep moving forward and avoid said obstacles.", #old: Describe the path ahead and any obstacles in exactly 5 words
                "image": img_base64
            }
            
            # 4. Request: Wait up to 20s for the VLM to generate text
            response = requests.post(self.vlm_endpoint, json=payload, timeout=20.0)
            
            if response.status_code == 200:
                res_text = response.json().get("response", "").strip()
                # Clear description if empty to let YOLO take back control
                self.vlm_description = res_text if res_text else "Empty Response"
            else:
                self.vlm_description = f"VLM Server Error: {response.status_code}"
            
            # Start cooldown timer to clear the message from the CSV
            threading.Timer(0.5, self.reset_vlm_status).start()

        except Exception as e:
            self.vlm_description = f"VLM Error: {str(e)}"
            threading.Timer(1.0, self.reset_vlm_status).start()
        finally:
            self.vlm_inference_duration = time.time() - start_time
            # CRITICAL: Unfreeze the pose_callback loop
            self.vlm_busy = False

    def pose_callback(self, msg: PoseStamped):
        t = msg.header.stamp
        p = msg.pose.position

        # --- SYSTEM FREEZE LOGIC ---
        if self.vlm_busy:
            self.csv_writer.writerow([
                t.sec, t.nanosec, p.x, p.y, p.z, 
                "PAUSED", "0.0", "VLM is thinking...", "", 0.0
            ])
            self.csv_file.flush()
            return 
        # ---------------------------

        if self.latest_frame is None:
            return

        # Use the frame caught by the capture loop
        frame = self.latest_frame.copy()
        
        # Process Depth
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            depth_frame = frames.get_depth_frame()
            if not depth_frame: return
            depth_image = np.asanyarray(depth_frame.get_data())
            roi = depth_image[200:280, 300:340]
            valid_depths = roi[roi > 0]
            min_depth_m = np.min(valid_depths) * 0.001 if valid_depths.size > 0 else 0.0
        except Exception:
            min_depth_m = 0.0

        # YOLO Inference
        results = self.yolo_model(frame, verbose=False)
        annotated_frame = results[0].plot()
        cv2.putText(annotated_frame, f"Obstacle: {round(min_depth_m, 2)}m", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # FIX: Put the annotated version back into the shared variable so the mp4 catches it
        self.latest_frame = annotated_frame 
        
        cv2.imshow("Go2 Live View", annotated_frame)
        cv2.waitKey(1)

        confidences = results[0].boxes.conf.tolist()
        max_conf = max(confidences) if confidences else 0.0
        detected = [results[0].names[int(c)] for c in results[0].boxes.cls]
        yolo_results_str = ", ".join(detected) if detected else "None"

        # Priority Logging Logic
        if max_conf >= 0.30:
            current_row_status = "Confidence High (YOLO)"
        elif 0.0 < max_conf < 0.30 and not self.vlm_busy:
            self.vlm_busy = True
            self.vlm_inference_duration = None
            current_row_status = "TRIGGERED: Starting VLM..."
            threading.Thread(target=self.vlm_request_worker, args=(frame,), daemon=True).start()
        elif self.vlm_description:
            current_row_status = f"VLM RESULT: {self.vlm_description}"
        else:
            current_row_status = "Waiting"

        duration_str = f"{self.vlm_inference_duration:.4f}" if self.vlm_inference_duration is not None else ""

        self.csv_writer.writerow([
            t.sec, t.nanosec, p.x, p.y, p.z, 
            yolo_results_str, f"{max_conf:.4f}", current_row_status, duration_str, round(min_depth_m, 2)
        ])
        self.csv_file.flush()

    def shutdown(self):
        self.get_logger().info("Shutting down node...")
        cv2.destroyAllWindows()
        self.pipeline.stop()
        self.video_writer.release()
        self.csv_file.close()
        rclpy.shutdown()
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