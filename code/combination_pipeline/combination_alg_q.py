import os
import cv2
import base64
import requests
import time
from pathlib import Path
import json
import pyrealsense2 as rs
import numpy as np
from ultralytics import YOLO

# Configuration
OUTPUT_DIR = "output_dir"
MODEL_NAME = "llava-13b"
PROMPT_TEMPLATE = """
You are a navigation safety assistant for a visually impaired user.
Based on this image of an object, say one of:
1. "[object] at [position], move [direction]"
2. "clear"
Do not explain. Only return one line.
"""

# Ensure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

class WalkableAreaDetector:
    def __init__(self, model_path="walkable_model1.pt", device='cpu'):
        # Set the device for the segmentation model
        self.device = device
        print(f"WalkableAreaDetector using device: {self.device}")
        try:
            self.model = YOLO(model_path)
            # Send the model to the specified device
            self.model.to(self.device) 
            print(f"Loaded walkable area model from {model_path}")
        except Exception as e:
            print(f"Error loading walkable model: {e}")
            self.model = None

    def detect(self, frame, conf=0.3):
        if self.model is None:
            return []
        try:
            # Use the specified device during inference
            results = self.model(frame, conf=conf, device=self.device, verbose=False)
            sidewalk_masks = []
            for result in results:
                if hasattr(result, 'masks') and result.masks is not None:
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        for i in range(len(result.boxes)):
                            class_id = int(result.boxes.cls[i])
                            confidence = float(result.boxes.conf[i])
                            # Only return masks for the classes you care about (e.g., 0=Road, 1=Sidewalk)
                            if class_id in [0, 1]: 
                                # Move mask data to CPU for NumPy/OpenCV processing
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
            # Resize mask to frame size (640x480 in this case)
            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST) 
            mask_bool = mask_resized > 0.5
            
            # Use BGR colors for OpenCV
            if class_id == 0:
                color = (0, 165, 255)  # Orange for Road
                label_prefix = "Road"
            elif class_id == 1:
                color = (0, 255, 0)    # Green for Sidewalk
                label_prefix = "Sidewalk"
            else:
                color = (255, 0, 0)
                label_prefix = f"Class {class_id}"
            
            # Apply color to the overlay where the mask is true
            overlay[mask_bool] = color
            
            # Add label in the center of the mask
            y_indices, x_indices = np.where(mask_bool)
            if len(x_indices) > 0 and len(y_indices) > 0:
                center_x = int(np.mean(x_indices))
                center_y = int(np.mean(y_indices))
                label = f"{label_prefix} {confidence:.2f}"
                # Draw the label text (using the original frame, not the overlay)
                cv2.putText(frame, label, (center_x - 50, center_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
        # Blend the overlay with the original frame (adjusting the weights for visibility)
        return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

def crop_and_encode(image, box, pad=20):
    x1, y1, x2, y2 = map(int, map(round, box))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(image.shape[1] - 1, x2 + pad)
    y2 = min(image.shape[0] - 1, y2 + pad)
    
    if x2 <= x1 or y2 <= y1:
        print("Invalid crop size, skipping.")
        return None

    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        print("Empty crop, skipping.")
        return None

    _, buffer = cv2.imencode('.jpg', cropped)
    return base64.b64encode(buffer).decode("utf-8")

def query_llava_base64(encoded_img, prompt):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "images": [encoded_img],
        "stream": False
    }
    try:
        res = requests.post("http://localhost:11434/api/chat", json=payload)
        if res.ok:
            data = res.text
            # Try to parse only the first JSON object if extra data is sent
            first_json = data.split('\n')[0]
            parsed = json.loads(first_json)
            return parsed["message"]["content"].strip().lower()
        else:
            print(f"Error querying LLava: {res.status_code}")
            return ""
    except Exception as e:
        print(f"Exception when querying LLava: {e}")
        return ""


def draw_danger_box(image, box, label):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(image, label, (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)



def process_camera_feed(output_video=None):
    """Process live RealSense camera feed with depth awareness and segmentation."""
    
    # Initialize RealSense pipeline (unchanged)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale:", depth_scale)

    # Initialize YOLO object detection model (using explicit CUDA as discussed)
    # NOTE: Ensure you have a YOLO model that supports **instance segmentation** (e.g., yolov8n-seg.pt) 
    # if you want a local model for object masks. yolov8n.pt is only for bounding box detection.
    model = YOLO('yolov8n.pt') 

    # 🔑 INITIALIZE THE WALKABLE AREA DETECTOR
    try:
        walkable_detector = WalkableAreaDetector(model_path="walkable_model1.pt", device='cuda:0')
    except Exception as e:
        print(f"Could not initialize WalkableAreaDetector: {e}. Running without segmentation.")
        walkable_detector = None
        
    # Initialize video writer (unchanged)
    out = None
    width, height = 640, 480
    if output_video:
        os.makedirs(os.path.dirname(output_video), exist_ok=True)
        out = cv2.VideoWriter(output_video,
                              cv2.VideoWriter_fourcc(*'mp4v'),
                              10,
                              (width, height))

    frame_count = 0
    try:
        while True:
            start_time = time.time()

            # Wait for frames (unchanged)
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            processed_frame = color_image.copy()
            
            # 🔑 STEP 1: Run Walkable Area Detection
            sidewalk_masks = []
            if walkable_detector:
                sidewalk_masks = walkable_detector.detect(color_image)
                # 🔑 STEP 2: Visualize the Walkable Area
                processed_frame = walkable_detector.visualize(processed_frame, sidewalk_masks)
                
            # Run YOLO object detection (using explicit CUDA as discussed)
            result = model(color_image, device='cpu')[0]

            # Get boxes and ensure they are pixel coordinates (unchanged)
            boxes = result.boxes.xyxy.cpu().numpy()
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, color_image.shape[1])
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, color_image.shape[0])

            threat_detected = False

            if result.boxes is not None:
                # ... (rest of the object detection and LLava logic remains here)
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)

                    # Compute center of the bounding box
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    # Get depth value at center
                    depth_in_meters = depth_image[center_y, center_x] * depth_scale
                    distance_ft = depth_in_meters * 3.28084

                    # Crop and send to llava
                    encoded_crop = crop_and_encode(color_image, box)
                    if encoded_crop is None:
                        continue 

                    response = query_llava_base64(encoded_crop, PROMPT_TEMPLATE)
                    print("🧠", response)


                    # Check if it's a hazard
                    if "move" in response or (depth_in_meters < 1.5 and depth_in_meters > 0):
                        threat_detected = True
                        draw_danger_box(processed_frame, box, f"{response} | {distance_ft:.1f} ft")
                    else:
                        label = f"{distance_ft:.1f} ft"
                        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(processed_frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # ... (end of object detection logic)

            # Display hazard or safe message (unchanged)
            status_text = "⚠️ Hazard detected" if threat_detected else "✅ No immediate hazards"
            color = (0, 0, 255) if threat_detected else (0, 255, 0)
            cv2.putText(processed_frame, status_text, (10, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Save snapshots (unchanged)
            if frame_count % 30 == 0:
                timestamp = int(time.time())
                snapshot_path = os.path.join(OUTPUT_DIR, f"frame_{timestamp}.jpg")
                cv2.imwrite(snapshot_path, processed_frame)
                print(f"Saved snapshot to {snapshot_path}")

            # Write to video (unchanged)
            if out:
                out.write(processed_frame)

            # Display frame (unchanged)
            cv2.imshow("Robot Navigation View (with Depth and Segmentation)", processed_frame)

            frame_count += 1

            # Print FPS (unchanged)
            elapsed_time = time.time() - start_time
            fps = 1.0 / elapsed_time if elapsed_time > 0 else 0
            print(f"Frame {frame_count} | FPS: {fps:.2f}")

            # Handle keyboard input (unchanged)
            key = cv2.waitKey(1) & 0xFF

            # ... (rest of keyboard input handling)
            if key == 27:  # ESC key to quit
                break
            elif key == ord('q'): 
                question = input("❓ Enter your question for the VLM: ")
                _, buffer = cv2.imencode('.jpg', color_image)
                encoded_img = base64.b64encode(buffer).decode("utf-8")
                payload = {
                    "model": MODEL_NAME,
                    "messages": [
                        {"role": "user", "content": question}
                    ],
                    "images": [encoded_img],
                    "stream": False
                }
                try:
                    res = requests.post("http://localhost:11434/api/chat", json=payload)
                    if res.ok:
                        data = res.text.split('\n')[0]
                        parsed = json.loads(data)
                        response = parsed["message"]["content"].strip()
                        print("🧠 VLM Response:", response)
                        cv2.putText(processed_frame, response, (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    else:
                        print(f"⚠️ VLM error: {res.status_code}")
                except Exception as e:
                    print(f"⚠️ VLM exception: {e}")


    except KeyboardInterrupt:
        print("Stopping RealSense video processing.")
    finally:
        pipeline.stop()
        if out:
            out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    output_video = os.path.join(OUTPUT_DIR, 'robot_navigation_feed2.mp4')
    process_camera_feed(output_video)
    print(f"Done. Output saved to {output_video}")