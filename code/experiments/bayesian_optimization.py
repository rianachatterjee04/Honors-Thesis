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
import gpytorch  
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.models.transforms import Normalize, Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan

# Fix for OpenMP conflict
os.environ['LD_PRELOAD'] = "/usr/lib/aarch64-linux-gnu/libgomp.so.1"

# --- BO UTILITY FUNCTIONS ---

def get_next_threshold(db_path="bo_history.csv"):
    """
    Calculates the next YOLO confidence threshold using Bayesian Optimization.
    Ensures numerical stability for Jetson/ARM64 hardware.
    """
    initial_points = [0.1, 0.3, 0.5, 0.75, 0.9]

    if not os.path.exists(db_path):
        return initial_points[0]

    try:
        df = pd.read_csv(db_path)
    except Exception:
        return initial_points[0]

    if len(df) < len(initial_points):
        return initial_points[len(df)]

    try:
        # Filter out zero-latency rows — VLM never triggered, not valid signal
        df = df[df['latency'] > 0] if 'latency' in df.columns else df
        if len(df) < 2:
            import random
            return round(random.uniform(0.2, 0.6), 2)
        # ---- Aggregate duplicates and build tensors ----
        df2 = df[['threshold', 'reward']].dropna()
        df2 = df2.groupby('threshold', as_index=False)['reward'].mean()

        train_x = torch.tensor(df2['threshold'].values, dtype=torch.double).unsqueeze(-1).clamp(0.0, 1.0)
        train_y = torch.tensor(df2['reward'].values, dtype=torch.double).unsqueeze(-1)

        gp = SingleTaskGP(
            train_x,
            train_y,
            input_transform=Normalize(d=1),
            outcome_transform=Standardize(m=1),
        )
        gp.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-3))

        with gpytorch.settings.cholesky_jitter(1e-2): 
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

            # IMPORTANT: eval mode before posterior/acq
            gp.eval()
            gp.likelihood.eval()

            # Debug sanity check
            test_x = torch.linspace(0.0, 1.0, 101, dtype=torch.double).unsqueeze(-1)
            post = gp.posterior(test_x)
            print(
                "[BO DEBUG] any NaN mean?", torch.isnan(post.mean).any().item(),
                "any NaN var?", torch.isnan(post.variance).any().item(),
                "min var:", float(post.variance.min().item())
            )
            noise = gp.likelihood.noise.item()
            print(f"[BO DEBUG] learned noise variance = {noise:.6g}, std = {(noise**0.5):.6g}")

            UCB = UpperConfidenceBound(gp, beta=1.0)

            try:
                new_x, _ = optimize_acqf(
                    UCB,
                    bounds=torch.tensor([[0.0], [1.0]], dtype=torch.double),
                    q=1,
                    num_restarts=5,
                    raw_samples=128,
                    options={"maxiter": 200},
                )
                return float(np.clip(new_x.item(), 0.0, 1.0))

            except Exception as e:
                print(f"BO Optimization Error: {e}. Using GRID fallback.")
                with torch.no_grad():
                    # FIXED SHAPE: (batch, q=1, d=1)
                    grid = torch.linspace(0.0, 1.0, 201, dtype=torch.double).view(-1, 1, 1)
                    vals = UCB(grid)
                    vals = torch.nan_to_num(vals, neginf=-1e9, posinf=1e9)
                    best_x = grid[torch.argmax(vals)]
                    return float(best_x.item())

    except Exception as e:
        print(f"BO Optimization Error: {e}. Using random fallback.")
        import random
        return round(random.uniform(0.1, 0.9), 2)

def save_trial_result(threshold, latency, usage, user_score=None, db_path="bo_history.csv"):
    clamped_latency_penalty = min(latency / 1000.0, 1.0)
    tech_penalty = -clamped_latency_penalty - (usage * 5.0)
    score_val = float(user_score) if user_score is not None else 5.0
    human_reward = (score_val / 2.0)
    reward = tech_penalty + human_reward

    new_data = pd.DataFrame([[threshold, latency, usage, score_val, reward]],
                            columns=['threshold', 'latency', 'usage', 'user_score', 'reward'])

    if not os.path.exists(db_path):
        new_data.to_csv(db_path, index=False, header=True)
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
        raw_video_filename = os.path.join(self.save_dir, f"trial_x_{self.conf_threshold}_raw.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(video_filename, fourcc, 15.0, (640, 480))
        self.raw_video_writer = cv2.VideoWriter(raw_video_filename, fourcc, 15.0, (640, 480))

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
        msg.point.x = 3.0
        self.wp_pub.publish(msg)

    def capture_loop(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=500)
            color_frame = frames.get_color_frame()
            if color_frame:
                self.latest_frame = np.asanyarray(color_frame.get_data())
        except:
            pass

    def vlm_request_worker(self, frame, stamp, pose_x):
        """Worker thread to handle the HTTP request and log result once received."""
        self.total_vlm_calls += 1
        start_time = time.time()
        try:
            small_frame = cv2.resize(frame, (320, 240))
            _, buffer = cv2.imencode('.jpg', small_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            payload = {"prompt": "Identify obstacles and hazards that a visually impaired person should look out for.", "image": img_base64}
            response = requests.post(self.vlm_endpoint, json=payload, timeout=30.0)

            if response.status_code == 200:
                self.vlm_description = response.json().get("response", "").strip().replace(",", ";")
                self.vlm_inference_duration = (time.time() - start_time) * 1000
                self.latencies.append(self.vlm_inference_duration)

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
            threading.Thread(target=self.vlm_request_worker,
                             args=(frame, msg.header.stamp.sec, msg.pose.position.x),
                             daemon=True).start()
            status = "VLM_TRIGGERED"
        else:
            status = "YOLO_ONLY"

        annotated_frame = results[0].plot()
        self.video_writer.write(annotated_frame)
        self.raw_video_writer.write(frame)  # unannotated — for offline BO trials

        p_x = self.total_vlm_calls / self.total_steps if self.total_steps > 0 else 0.0
        self.csv_writer.writerow([msg.header.stamp.sec, msg.pose.position.x, 0.0, 0.0,
                                  max_conf, status, "PENDING...", 0.0, p_x])
        self.csv_file.flush()

    def shutdown_timer_callback(self):
        while self.vlm_busy:
            self.get_logger().info("Waiting for final VLM response before closing...")
            time.sleep(0.5)

        self.pipeline.stop()
        self.video_writer.release()
        self.raw_video_writer.release()
        self.csv_file.close()

        avg_latency = np.mean(self.latencies) if self.latencies else 0.0
        usage_ratio = self.total_vlm_calls / self.total_steps if self.total_steps > 0 else 0.0

        user_input = None
        db_path = "bo_history.csv"

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

        save_trial_result(self.conf_threshold, avg_latency, usage_ratio, user_score=user_input)

        self.get_logger().info(f"TRIAL COMPLETE. Latency: {avg_latency:.2f}, Usage: {usage_ratio:.4f}")
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