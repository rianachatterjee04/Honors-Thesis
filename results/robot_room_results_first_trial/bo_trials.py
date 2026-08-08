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

# --- BO UTILITY FUNCTIONS ---

def get_next_threshold(db_path="bo_history.csv"):
    """
    Calculates the next YOLO confidence threshold using Bayesian Optimization.
    Ensures numerical stability for Jetson/ARM64 hardware.
    """
    initial_points = [0.1, 0.5, 0.9]
    
    if not os.path.exists(db_path):
        return initial_points[0]

    try:
        df = pd.read_csv(db_path)
    except Exception:
        return initial_points[0]

    if len(df) < len(initial_points):
        return initial_points[len(df)]

    try:
        train_x = torch.tensor(df['threshold'].values, dtype=torch.double).unsqueeze(-1)
        train_y = torch.tensor(df['reward'].values, dtype=torch.double).unsqueeze(-1)

        if train_y.std() > 1e-4:
            train_y = (train_y - train_y.mean()) / train_y.std()
        else:
            train_y = train_y - train_y.mean()

        from gpytorch.constraints import GreaterThan
        gp = SingleTaskGP(train_x, train_y)
        gp.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-4))
        
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        UCB = UpperConfidenceBound(gp, beta=0.5)
        
        new_x, _ = optimize_acqf(
            UCB, 
            bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double), 
            q=1, 
            num_restarts=10, 
            raw_samples=512
        )
        
        return float(np.clip(new_x.item(), 0.0, 1.0))

    except Exception as e:
        print(f"BO Optimization Error: {e}. Using random fallback.")
        import random
        return round(random.uniform(0.1, 0.9), 2)

def save_trial_result(threshold, latency, usage, user_score=None, db_path="bo_history.csv"):
    tech_penalty = -(latency / 1000.0) - (usage * 2.0)
    score_val = float(user_score) if user_score is not None else 5.0
    human_reward = (score_val / 2.0) 
    reward = tech_penalty + human_reward
    
    new_data = pd.DataFrame([[threshold, latency, usage, score_val, reward]], 
                            columns=['threshold', 'latency', 'usage', 'user_score', 'reward'])
    
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
        
        self.save_dir = "/home/unitree/GenAssist_Riana/in_line_test/"
        os.makedirs(self.save_dir, exist_ok=True)
        
        video_filename = os.path.join(self.save_dir, f"trial_x_{self.conf_threshold}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(video_filename, fourcc, 15.0, (640, 480))

        self.wp_pub = self.create_publisher(PointStamped, '/way_point', 10)
        self.pose_sub = self.create_subscription(PoseStamped, '/utlidar/robot_pose', self.pose_callback, 10)

        self.csv_path = os.path.join(self.save_dir, f"bo_trial_x_{self.conf_threshold}.csv")
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
        self.create_timer(20.0, self.shutdown_timer_callback) 

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

    def vlm_request_worker(self, frame, stamp, pose_x):
        """Worker thread to handle the HTTP request and log result once received."""
        self.total_vlm_calls += 1 
        start_time = time.time() 
        try:
            small_frame = cv2.resize(frame, (320, 240)) 
            _, buffer = cv2.imencode('.jpg', small_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            payload = {"prompt": "Describe the path ahead and any obstacles that a visually impaired person should be aware of.", "image": img_base64}	#old: Identify hazards.
            response = requests.post(self.vlm_endpoint, json=payload, timeout=30.0)
            
            if response.status_code == 200:
                self.vlm_description = response.json().get("response", "").strip().replace(",", ";")
                self.vlm_inference_duration = (time.time() - start_time) * 1000 
                self.latencies.append(self.vlm_inference_duration)
                
                # ASYNC LOGGING: Write the actual result here once it exists
                p_x = self.total_vlm_calls / self.total_steps if self.total_steps > 0 else 0.0
                self.csv_writer.writerow([stamp, pose_x, 0.0, 0.0, "N/A", "VLM_RESPONSE_RECEIVED", 
                                          self.vlm_description, self.vlm_inference_duration, p_x])
                self.csv_file.flush()
        except Exception as e:
            print(f"VLM Worker Thread Error: {e}")
        finally:
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
            # Pass current pose data to worker so the log matches the trigger location
            threading.Thread(target=self.vlm_request_worker, 
                             args=(frame, msg.header.stamp.sec, msg.pose.position.x), 
                             daemon=True).start()
            status = "VLM_TRIGGERED"
        else:
            status = "YOLO_ONLY"

        annotated_frame = results[0].plot()
        self.video_writer.write(annotated_frame)

        p_x = self.total_vlm_calls / self.total_steps if self.total_steps > 0 else 0.0
        # This row captures the YOLO state immediately
        self.csv_writer.writerow([msg.header.stamp.sec, msg.pose.position.x, 0.0, 0.0, 
                                  max_conf, status, "PENDING...", 0.0, p_x])
        self.csv_file.flush()

    def shutdown_timer_callback(self):
        """
        Safely shuts down the node, ensuring all asynchronous VLM 
        responses are recorded before closing the logging files.
        """
        self.get_logger().info("Shutting down trial...")

        # 1. Wait for the final VLM worker thread to finish writing to the CSV
        # This prevents the 'I/O operation on closed file' error.
        wait_start = time.time()
        while self.vlm_busy:
            if time.time() - wait_start > 10.0:  # 10-second safety timeout
                self.get_logger().warn("VLM worker timed out during shutdown. Closing anyway.")
                break
            self.get_logger().info("Waiting for final VLM response to be logged...")
            time.sleep(0.5)

        # 2. Stop hardware and release file handles
        self.pipeline.stop()
        self.video_writer.release()
        self.csv_file.close()
        
        # 3. Calculate trial metrics for the Bayesian Optimization reward
        avg_latency = np.mean(self.latencies) if self.latencies else 0.0
        usage_ratio = self.total_vlm_calls / self.total_steps if self.total_steps > 0 else 0.0
        
        # 4. Human-in-the-Loop (HITL) Feedback
        user_input = None
        db_path = "bo_history.csv"
        
        # Only ask for feedback if it's a legitimate trial (not the first one)
        if os.path.exists(db_path):
            print("\n" + "="*40)
            print("TRIAL ENDED: HUMAN FEEDBACK REQUIRED")
            while True:
                try:
                    user_input = input("Rate user response to guidance (1=Fail/Unsafe, 10=Excellent/Smooth): ")
                    val = float(user_input)
                    if 1 <= val <= 10: break
                    print("Please enter a number between 1 and 10.")
                except ValueError:
                    print("Invalid input. Enter a number.")
            print("="*40 + "\n")

        # 5. Log the final result to the BO database
        save_trial_result(self.conf_threshold, avg_latency, usage_ratio, user_score=user_input)
        
        self.get_logger().info(f"TRIAL COMPLETE. Avg Latency: {avg_latency:.2f}ms, Usage: {usage_ratio:.4f}")
        
        # 6. Exit the script
        rclpy.shutdown()
        os._exit(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=None)
    args, _ = parser.parse_known_args()

    if args.threshold is None:
        selected_threshold = get_next_threshold()
        print(f"\n[BO ADVICE] Next suggested threshold: {selected_threshold:.4f}\n")
    else:
        selected_threshold = args.threshold

    rclpy.init()
    node = Go2WalkForward(threshold=selected_threshold)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.shutdown_timer_callback()

if __name__ == '__main__':
    main()
