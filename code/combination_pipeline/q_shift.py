import os
import cv2
import time
import torch
import numpy as np
import time 
import base64
import requests
import tempfile
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
import json  # needed for parsing VLM responses
import speech_recognition as sr

def listen_for_voice(prompt="Say your command..."):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print(prompt)
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio)
        print(f"You said: {text}")
        return text
    except sr.UnknownValueError:
        print("Could not understand audio.")
    except sr.RequestError:
        print("Speech recognition service unavailable.")
    return None


# Configuration constants
OUTPUT_DIR = "output_dir"
DISTANCE_THRESHOLD_FT = 15.0  # Objects within this many feet trigger warnings
DISPLAY_SCALE = 1.0  # Scale display for high-res screens
SAVE_FRAMES = True  # Save frames for debugging/demo
SHOW_DEPTHS = True  # Show depth information
VLM_ENABLED = True  # Enable VLM direction queries
MODEL_NAME = "llava-phi3:latest"  # VLM model to query

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


class WalkableAreaDetector:
    def __init__(self, model_path="walkable_model1.pt"):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Apple Silicon GPU via MPS ✅")
        else:
            self.device = torch.device("cpu")
            print("⚠️ No GPU found, using CPU")


        print(f"WalkableAreaDetector using device: {self.device}")
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
        
            print(f"Loaded walkable area model from {model_path}")
        except Exception as e:
            print(f"Error loading walkable model: {e}")
            self.model = None

    def detect(self, frame, conf=0.3):
        if self.model is None:
            return []
        try:
            results = self.model(frame, conf=conf, verbose=False)
            sidewalk_masks = []
            for result in results:
                if hasattr(result, 'masks') and result.masks is not None:
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        for i in range(len(result.boxes)):
                            class_id = int(result.boxes.cls[i])
                            confidence = float(result.boxes.conf[i])
                            print(f"Detected class: {class_id}, confidence: {confidence:.2f}")
                            if class_id in [0, 1]:
                                mask = result.masks.data[i].cpu().numpy()
                                sidewalk_masks.append((mask, confidence, class_id))
            return sidewalk_masks
        except Exception as e:
            print(f"Error in walkable area detection: {e}")
            return []

    def visualize(self, frame, sidewalk_masks):
        if not sidewalk_masks:
            return frame
        overlay = np.zeros_like(frame)
        for mask, confidence, class_id in sidewalk_masks:
            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            mask_bool = mask_resized > 0.5
            if class_id == 0:
                color = (0, 255, 255)
                label_prefix = "Road"
            elif class_id == 1:
                color = (0, 255, 0)
                label_prefix = "Side walk"
            else:
                color = (255, 0, 0)
                label_prefix = f"Class {class_id}"
            overlay[mask_bool] = color
            y_indices, x_indices = np.where(mask_bool)
            if len(x_indices) > 0 and len(y_indices) > 0:
                center_x = int(np.mean(x_indices))
                center_y = int(np.mean(y_indices))
                label = f"{label_prefix} {confidence:.2f}"
                cv2.putText(frame, label, (center_x-50, center_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)


class DepthEstimator:
    def __init__(self):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        print(f"DepthEstimator using device: {self.device}")
        self.model_type = "MiDaS_small"
        midas_path = "midas_model"
        os.makedirs(midas_path, exist_ok=True)
        model_path = os.path.join(midas_path, f"{self.model_type}.pt")
        if not os.path.exists(model_path):
            print(f"Downloading MiDaS model to {model_path}...")
            url = f"https://github.com/intel-isl/MiDaS/releases/download/v2_1/{self.model_type}.pt"
            os.system(f"curl -L -o {model_path} {url}")
        self.model = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.model.to(self.device)
        self.model.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.small_transform

    def estimate_depth(self, frame):
        input_batch = self.transform(frame).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth_map = prediction.cpu().numpy()
        depth_min, depth_max = depth_map.min(), depth_map.max()
        normalized_depth = 255 * (depth_map - depth_min) / (depth_max - depth_min)
        depth_colormap = cv2.applyColorMap(normalized_depth.astype("uint8"), cv2.COLORMAP_INFERNO)
        return depth_map, depth_colormap

    def get_depth_at_box(self, depth_map, box, assumed_max_distance_m=10.0):
        if isinstance(box, torch.Tensor):
            box = box.cpu().numpy()
        x1, y1, x2, y2 = map(int, map(round, box))
        y1 = max(0, min(y1, depth_map.shape[0] - 1))
        y2 = max(0, min(y2, depth_map.shape[0] - 1))
        x1 = max(0, min(x1, depth_map.shape[1] - 1))
        x2 = max(0, min(x2, depth_map.shape[1] - 1))
        if x2 <= x1 or y2 <= y1:
            return 100.0
        depth_values = depth_map[y1:y2, x1:x2].flatten()
        depth_values = depth_values[~np.isnan(depth_values)]
        depth_values = depth_values[depth_values > 0]
        if len(depth_values) == 0:
            return 100.0
        depth_min, depth_max = depth_map.min(), depth_map.max()
        normalized = (depth_values - depth_min) / (depth_max - depth_min + 1e-6)
        inverted = 1.0 - normalized
        median_relative = np.median(inverted)
        depth_in_meters = median_relative * assumed_max_distance_m
        return depth_in_meters


class ObjectDetector:
    def __init__(self, model="yolov8n.pt"):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        print(f"ObjectDetector using device: {self.device}")
        self.model = YOLO(model)
        self.model.to(self.device)
        self.model = YOLO(model)
        self.model.to(self.device)

    def detect(self, frame, conf=0.25):
        results = self.model(frame, conf=conf, verbose=False)[0]
        return results


def process_video(video_path, output_path=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_path}")
        return
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"Video dimensions: {width}x{height}, FPS: {fps}")

    walkable_detector = WalkableAreaDetector()
    depth_estimator = DepthEstimator()
    object_detector = ObjectDetector()

    out = None
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        out = cv2.VideoWriter(output_path,
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              fps, (width * 2, height))

    frame_count = 0
    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            display_frame = frame.copy()

            # Step 1: Walkable areas
            walkable_areas = walkable_detector.detect(frame)
            display_frame = walkable_detector.visualize(display_frame, walkable_areas)

            # Step 2: Depth
            depth_map, depth_colormap = depth_estimator.estimate_depth(frame)

            # Step 3: Objects
            results = object_detector.detect(frame)

            warnings = []
            if hasattr(results, 'boxes') and results.boxes is not None:
                for i, box in enumerate(results.boxes.xyxy):
                    x1, y1, x2, y2 = map(int, box[:4])
                    confidence = results.boxes.conf[i].item()
                    class_id = int(results.boxes.cls[i].item())
                    object_name = results.names[class_id]
                    center_x = (x1 + x2) / 2
                    depth_m = depth_estimator.get_depth_at_box(depth_map, box, 8)
                    depth_ft = depth_m * 3.28084
                    is_close = depth_ft < DISTANCE_THRESHOLD_FT
                    direction = "right" if center_x < width/2 else "left"
                    if is_close and VLM_ENABLED:
                        warnings.append(f"{object_name} detected, move {direction}")
                    color = (0, 0, 255) if is_close else (0, 255, 0)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{object_name}: {depth_ft:.1f}ft"
                    if is_close:
                        label += f" | {object_name} ahead, move {direction}"
                    cv2.putText(display_frame, label, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if warnings:
                print(f"Warnings: {', '.join(warnings)}")

            depth_resized = cv2.resize(depth_colormap, (width, height))
            combined_view = np.hstack((display_frame, depth_resized))

            if out:
                out.write(combined_view)

            if SHOW_DEPTHS:
                if DISPLAY_SCALE != 1.0:
                    display_w = int(width * 2 * DISPLAY_SCALE)
                    display_h = int(height * DISPLAY_SCALE)
                    combined_display = cv2.resize(combined_view, (display_w, display_h))
                    cv2.imshow("Navigation Assistant", combined_display)
                else:
                    cv2.imshow("Navigation Assistant", combined_view)
            else:
                cv2.imshow("Navigation Assistant", display_frame)

            frame_count += 1
            elapsed = time.time() - start_time
            fps_current = 1.0 / elapsed if elapsed > 0 else 0
            print(f"Frame {frame_count} | FPS: {fps_current:.2f}")

            # Handle key input (ESC / Q-shift)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('q'):
                vlm_frame = cv2.resize(frame, (640, 360))
                question = listen_for_voice("🎙️ Ask your question for the VLM:")
                if not question:
                    continue
                _, buffer = cv2.imencode('.jpg', vlm_frame)
                encoded_img = base64.b64encode(buffer).decode("utf-8")
                payload = {
                    "model": MODEL_NAME,
                    "prompt": question,
                    "images": [encoded_img],
                    "stream": False
                }
                try:
                    # start timer
                    t0 = time.time()

                    res = requests.post("http://localhost:11434/api/generate", json=payload)

                    # stop timer
                    t1 = time.time()
                    latency_ms = (t1 - t0) * 1000.0  # convert to milliseconds

                    if res.ok:
                        data = res.text.split('\n')[0]
                        parsed = json.loads(data)
                        response = parsed.get("response", "").strip()
                        if not response:
                            response = "[No response generated]"
                        response_one_line = " ".join(response.split())

                        print(f"🧠 VLM Response: {response_one_line}")
                        print(f"⏱️ VLM latency: {latency_ms:.2f} ms")

                        # Overlay response + latency on frame
                        cv2.putText(display_frame, response, (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        cv2.putText(display_frame, f"{latency_ms:.1f} ms", (20, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    else:
                        print(f"⚠️ VLM error: {res.status_code}")
                except Exception as e:
                    print(f"⚠️ VLM exception: {e}")

    except KeyboardInterrupt:
        print("Processing stopped by user")
    finally:
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        print("Processing complete")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "YOLO_VLM_switch_off/videos/clip_20_to_107.mp4"
    video_name = os.path.basename(video_path).split('.')[0]
    output_path = os.path.join(OUTPUT_DIR, f"{video_name}_processed.mp4")
    process_video(video_path, output_path)
    print(f"Output saved to {output_path}")
