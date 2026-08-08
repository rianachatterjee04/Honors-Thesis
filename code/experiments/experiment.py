#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np
import torch
import time
import base64
import requests

from ultralytics import YOLO

# ----------------------------
# CONFIG
# ----------------------------

TARGET_DISTANCE_M = 1.83       # ~6 feet
FORWARD_SPEED = 0.25           # m/s (safe walking pace)
TURN_SPEED = 0.4               # rad/s
SIDE_STEP_TIME = 1.2           # seconds
STOP_DISTANCE_M = 1.2          # obstacle threshold
YOLO_CONF = 0.01               # VERY LOW confidence
DEPTH_FOV_RATIO = 0.3          # center depth window
OLLAMA_ENDPOINT = "http://localhost:5000/vlm"
VLM_URL = "http://localhost:11434/api/generate"
VLM_MODEL = "llava-phi3"

# ----------------------------
# NODE
# ----------------------------

class Go2Navigator(Node):
    def __init__(self):
        super().__init__("go2_yolo_vlm_nav")

        self.bridge = CvBridge()

        # Subscribers
        self.rgb_sub = self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.rgb_callback,
            10
        )

        self.depth_sub = self.create_subscription(
            Image,
            "/camera/depth/image_raw",
            self.depth_callback,
            10
        )

        # Publisher
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # Models
        self.yolo = YOLO("yolov8n.pt")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo.to(self.device)

        # State
        self.rgb_frame = None
        self.depth_frame = None
        self.start_time = None
        self.distance_traveled = 0.0
        self.last_cmd_time = time.time()

        self.get_logger().info("Go2 YOLO + VLM Navigator Initialized")

    # ----------------------------
    # CALLBACKS
    # ----------------------------

    def rgb_callback(self, msg):
        self.rgb_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def depth_callback(self, msg):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        self.depth_frame = depth.astype(np.float32) / 1000.0  # mm → meters

    # ----------------------------
    # CORE LOGIC
    # ----------------------------

    def get_forward_depth(self):
        """Return minimum depth in center of frame"""
        if self.depth_frame is None:
            return np.inf

        h, w = self.depth_frame.shape
        cx1 = int(w * (0.5 - DEPTH_FOV_RATIO / 2))
        cx2 = int(w * (0.5 + DEPTH_FOV_RATIO / 2))
        cy1 = int(h * 0.4)
        cy2 = int(h * 0.8)

        roi = self.depth_frame[cy1:cy2, cx1:cx2]
        roi = roi[roi > 0]

        if roi.size == 0:
            return np.inf

        return np.percentile(roi, 10)  # robust minimum

    def query_vlm(self, image):
        """
        Send RGB frame to local Flask VLM server (port 5000)
        Returns: 'left', 'right', 'forward', or 'stop'
        """

        # Encode image
        _, buffer = cv2.imencode(".jpg", image)
        image_b64 = base64.b64encode(buffer).decode("utf-8")

        prompt = (
            "You are assisting a blind user using a robot dog. "
            "Based on the image, determine if there is an obstacle directly ahead. "
            "Respond with ONE word only: left, right, forward, or stop."
        )

        payload = {
            "image": image_b64,
            "prompt": prompt
        }

        try:
            r = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=5)

            if not r.ok:
                return "stop"

            response = r.json().get("response", "").lower()

            # Hard safety filter
            if "left" in response:
                return "left"
            elif "right" in response:
                return "right"
            elif "forward" in response:
                return "forward"
            else:
                return "stop"

        except Exception as e:
            self.get_logger().warn(f"VLM error: {e}")
            return "stop"


    def step(self):
        if self.rgb_frame is None or self.depth_frame is None:
            return

        if self.start_time is None:
            self.start_time = time.time()

        elapsed = time.time() - self.start_time
        self.distance_traveled = elapsed * FORWARD_SPEED

        if self.distance_traveled >= TARGET_DISTANCE_M:
            self.stop()
            self.get_logger().info("Reached target distance")
            rclpy.shutdown()
            return

        depth_ahead = self.get_forward_depth()

        # YOLO inference
        results = self.yolo(self.rgb_frame, conf=YOLO_CONF, verbose=False)[0]

        obstacle_detected = False
        direction = "forward"

        if depth_ahead < STOP_DISTANCE_M:
            obstacle_detected = True

        if obstacle_detected:
            if results.boxes is not None and len(results.boxes) > 0:
                box = results.boxes.xyxy[0].cpu().numpy()
                cx = (box[0] + box[2]) / 2

                if cx < self.rgb_frame.shape[1] / 2:
                    direction = "right"
                else:
                    direction = "left"
            else:
                direction = self.query_vlm(self.rgb_frame)

        self.execute_motion(direction)

    # ----------------------------
    # MOTION COMMANDS
    # ----------------------------

    def execute_motion(self, direction):
        cmd = Twist()

        if direction == "forward":
            cmd.linear.x = FORWARD_SPEED

        elif direction == "left":
            cmd.angular.z = TURN_SPEED

        elif direction == "right":
            cmd.angular.z = -TURN_SPEED

        else:  # stop
            pass

        self.cmd_pub.publish(cmd)

    def stop(self):
        self.cmd_pub.publish(Twist())


# ----------------------------
# MAIN
# ----------------------------

def main():
    rclpy.init()
    node = Go2Navigator()

    rate = node.create_rate(10)

    try:
        while rclpy.ok():
            rclpy.spin_once(node)
            node.step()
            rate.sleep()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
