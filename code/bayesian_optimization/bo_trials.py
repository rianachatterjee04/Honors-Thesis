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
import argparse

# BO Specific Imports
import torch
import pandas as pd
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

# Fix for OpenMP conflict
os.environ['LD_PRELOAD'] = "/usr/lib/aarch64-linux-gnu/libgomp.so.1"

# --- BO UTILITY FUNCTIONS (OUTSIDE CLASS) ---

def get_next_threshold(db_path="bo_history.csv"):
    """Reads history from CSV and suggests the next threshold to try."""
    if not os.path.exists(db_path):
        initial_points = [0.1, 0.5, 0.9]
        return initial_points[0]

    df = pd.read_csv(db_path)
    initial_points = [0.1, 0.5, 0.9]
    if len(df) < len(initial_points):
        return initial_points[len(df)]

    # Prepare data for BoTorch
    train_x = torch.tensor(df['threshold'].values, dtype=torch.double).unsqueeze(-1)
    train_y = torch.tensor(df['reward'].values, dtype=torch.double).unsqueeze(-1)

    # Standardize/Fit GP
    gp = SingleTaskGP(train_x, train_y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    # UCB Acquisition Function
    UCB = UpperConfidenceBound(gp, beta=0.1)
    new_x, _ = optimize_acqf(
        UCB, bounds=torch.tensor([[0.0], [1.0]]), q=1, num_restarts=5, raw_samples=20
    )
    return float(new_x.item())

def save_trial_result(threshold, latency, usage, db_path="bo_history.csv"):
    """Calculates reward and saves it to the permanent history file."""
    # Custom Reward: Adjust weights to your preference
    reward = -(latency / 1000.0) - (usage * 5.0)
    
    new_data = pd.DataFrame([[threshold, latency, usage, reward]], 
                            columns=['threshold', 'latency', 'usage', 'reward'])
    
    if not os.path.exists(db_path):
        new_data.to_csv(db_path, index=False)
    else:
        new_data.to_csv(db_path, mode='a', header=False, index=False)

# --- ROBOT NODE ---

class Go2WalkForward(Node):
    def __init__(self, threshold):
        super().__init__('go2_walk_forward')
        self.conf_threshold = threshold
        self.get_logger().info(f"STARTING TRIAL: THRESHOLD {self.conf_threshold}")
        
        self.total_vlm_calls = 0        
        self.total_steps = 0            
        self.latencies = []             
        self.yolo_model = YOLO('yolov8n.pt')
        self.vlm_endpoint = "http://localhost:5000/vlm"
        self.vlm_busy = False
        self.vlm_description = "N/A"
        self.vlm_inference_duration = 0.0  
        self.latest_frame = None
        
        save_dir = "/home/unitree/GenAssist_Riana/in_line_test/"
        os.makedirs(save_dir, exist_ok=True)
        
        video_filename = os.path.join(save_dir, f"trial_x_{self.conf_threshold}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(video_filename, fourcc, 15.0, (640, 480))

        self.wp_pub = self.create_publisher(PointStamped, '/way_point', 10)
        self.pose_sub = self.create_subscription(PoseStamped, '/utlidar/robot_pose', self.pose_callback, 10)

        self.csv_path = os.path.join(save_dir, f"bo_trial_x_{self.conf_threshold}.csv")
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["stamp_sec", "x", "y", "z", "yolo_conf", "status", "vlm_response", "latency_ms", "usage_ratio"])

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
        
        try:
            self.pipeline.start(config)
        except Exception as e:
            self.get_logger().error(f"Hardware Error: {e}")

        self.create_timer(0.06, self.capture_loop)
        self.move_timer = self.create_timer(1.0, self.publish_waypoint)
        self.create_timer(20.0, self.shutdown) 

    def publish_waypoint(self):
        if self.vlm_busy: return
        msg = PointStamped()
        msg.header.frame_id = 'map'
        msg.point.x = 2.0 
        self.wp_pub.publish(msg)
        
    def capture_loop(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=500)
            color_frame = frames.get_color_frame()
            if color_frame:
                self.latest_frame = np.asanyarray(color_frame.get_data())
        except: pass

    def vlm_request_worker(self, frame):
        self.total_vlm_calls += 1 
        start_time = time.time() 
        try:
            small_frame = cv2.resize(frame, (320, 240)) 
            _, buffer = cv2.imencode('.jpg', small_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            payload = {"prompt": "Identify hazards.", "image": img_base64}
            response = requests.post(self.vlm_endpoint, json=payload, timeout=5.0)
            if response.status_code == 200:
                self.vlm_description = response.json().get("response", "").strip().replace(",", ";")
        except: pass
        finally:
            self.vlm_inference_duration = (time.time() - start_time) * 1000 
            self.latencies.append(self.vlm_inference_duration)
            self.vlm_busy = False

    def pose_callback(self, msg: PoseStamped):
        self.total_steps += 1 
        if self.latest_frame is None: return
        frame = self.latest_frame.copy()
        results = self.yolo_model(frame, verbose=False)
        confidences = results[0].boxes.conf.tolist()
        max_conf = max(confidences) if confidences else 0.0

        if max_conf < self.conf_threshold and not self.vlm_busy:
            self.vlm_busy = True
            threading.Thread(target=self.vlm_request_worker, args=(frame,), daemon=True).start()
            status = "VLM_TRIGGERED"
        else:
            status = "YOLO_ONLY"

        annotated_frame = results[0].plot()
        self.video_writer.write(annotated_frame)

        p_x = self.total_vlm_calls / self.total_steps if self.total_steps > 0 else 0.0
        self.csv_writer.writerow([msg.header.stamp.sec, msg.pose.position.x, 0.0, 0.0, max_conf, status, self.vlm_description, self.vlm_inference_duration, p_x])

    def shutdown(self):
        avg_latency = np.mean(self.latencies) if self.latencies else 0.0
        usage_ratio = self.total_vlm_calls / self.total_steps if self.total_steps > 0 else 0.0
        
        # LOG RESULTS TO THE BO DATABASE
        save_trial_result(self.conf_threshold, avg_latency, usage_ratio)
        
        self.get_logger().info(f"TRIAL COMPLETE. Latency: {avg_latency:.2f}, Usage: {usage_ratio:.4f}")
        self.pipeline.stop()
        self.video_writer.release()
        self.csv_file.close()
        rclpy.shutdown()
        os._exit(0)

def main():
    parser = argparse.ArgumentParser()
    # If you run without --threshold, it asks the BO brain for the best next value
    parser.add_argument('--threshold', type=float, default=None)
    args, _ = parser.parse_known_args()

    # Determine threshold automatically if not provided manually
    if args.threshold is None:
        selected_threshold = get_next_threshold()
        print(f"\n[BO ADVICE] Next suggested threshold based on history: {selected_threshold:.4f}\n")
    else:
        selected_threshold = args.threshold

    rclpy.init()
    node = Go2WalkForward(threshold=selected_threshold)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.shutdown()

if __name__ == '__main__':
    main()