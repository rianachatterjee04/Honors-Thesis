import os
import cv2
import time
import torch
import numpy as np
import base64
import requests
import tempfile
from pathlib import Path
import csv
from PIL import Image
from ultralytics import YOLO

# Configuration constants
OUTPUT_DIR = "output_dir"
DISTANCE_THRESHOLD_FT = 15.0  # Objects within this many feet trigger warnings
DISPLAY_SCALE = 1.0  # Scale display for high-res screens
SAVE_FRAMES = True  # Save frames for debugging/demo
SHOW_DEPTHS = True  # Show depth information
VLM_ENABLED = True  # Enable VLM direction queries

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

class WalkableAreaDetector:
    def __init__(self, model_path="walkable_model1.pt"):
        """Initialize walkable area detection model"""
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"WalkableAreaDetector using device: {self.device}")
        
        # Load the model
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            print(f"Loaded walkable area model from {model_path}")
        except Exception as e:
            print(f"Error loading walkable model: {e}")
            self.model = None
    
    def detect(self, frame, conf=0.3):
        """Detect sidewalks in a frame - with debugging"""
        if self.model is None:
            return []
        
        try:
            results = self.model(frame, conf=conf, verbose=False)
            sidewalk_masks = []
            
            for result in results:
                # Check if segmentation masks exist
                if hasattr(result, 'masks') and result.masks is not None:
                    # Get class information
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        for i in range(len(result.boxes)):
                            class_id = int(result.boxes.cls[i])
                            confidence = float(result.boxes.conf[i])
                            
                            # Debug: Print all detections
                            print(f"Detected class: {class_id}, confidence: {confidence:.2f}")
                            
                            # Try both class 0 (sidewalk) and class 1 (road) to see which looks right
                            if class_id in [0, 1]:  # Check both sidewalk and road classes
                                # Get the segmentation mask for this detection
                                mask = result.masks.data[i].cpu().numpy()
                                
                                # Add class ID to help debug
                                sidewalk_masks.append((mask, confidence, class_id))
            
            return sidewalk_masks
        except Exception as e:
            print(f"Error in walkable area detection: {e}")
            return []
            
    def visualize(self, frame, sidewalk_masks):
        """Render sidewalk segmentation masks on the frame - with class info"""
        if not sidewalk_masks:
            return frame

        # Create overlay for green mask
        overlay = np.zeros_like(frame)
        
        for mask, confidence, class_id in sidewalk_masks:
            # Convert binary mask to 3-channel
            mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            mask_bool = mask_resized > 0.5
            
            # Use different colors for different classes to debug
            if class_id == 0:  # Sidewalk
                color = (0, 255, 255)   # Green
                label_prefix = "Road"
            elif class_id == 1:  # Road
                color = (0, 255, 0) # Yellow for debugging
                label_prefix = "Side walk"
            else:
                color = (255, 0, 0)  # Blue for any other class
                label_prefix = f"Class {class_id}"
            
            # Apply color where mask is True
            overlay[mask_bool] = color
            
            # Add confidence text (find center of mask)
            y_indices, x_indices = np.where(mask_bool)
            if len(x_indices) > 0 and len(y_indices) > 0:
                center_x = int(np.mean(x_indices))
                center_y = int(np.mean(y_indices))
                label = f"{label_prefix} {confidence:.2f}"
                cv2.putText(frame, label, (center_x-50, center_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Blend overlay with original frame
        return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)


class DepthEstimator:
    def __init__(self):
        """Initialize MiDaS depth estimation model"""
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"DepthEstimator using device: {self.device}")
        
        # MiDaS model configuration
        self.model_type = "MiDaS_small"
        midas_path = "midas_model"
        os.makedirs(midas_path, exist_ok=True)
        model_path = os.path.join(midas_path, f"{self.model_type}.pt")
        
        # Download model if not available
        if not os.path.exists(model_path):
            print(f"Downloading MiDaS model to {model_path}...")
            url = f"https://github.com/intel-isl/MiDaS/releases/download/v2_1/{self.model_type}.pt"
            os.system(f"curl -L -o {model_path} {url}")
        
        # Load model
        self.model = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.model.to(self.device)
        self.model.eval()
        
        # Define transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.small_transform
    
    def estimate_depth(self, frame):
        """Estimate depth from a frame"""
        # Transform input for MiDaS
        input_batch = self.transform(frame).to(self.device)
        
        # Run inference
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        # Convert to numpy array
        depth_map = prediction.cpu().numpy()
        
        # Normalize depth map for visualization
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        normalized_depth = 255 * (depth_map - depth_min) / (depth_max - depth_min)
        depth_colormap = cv2.applyColorMap(normalized_depth.astype("uint8"), cv2.COLORMAP_INFERNO)
        
        return depth_map, depth_colormap
    
    def get_depth_at_box(self, depth_map, box, assumed_max_distance_m=10.0):
        # Convert box to numpy if it's a Tensor
        if isinstance(box, torch.Tensor):
            box = box.cpu().numpy()

        # Round and convert to integers
        x1, y1, x2, y2 = map(int, map(round, box))

        # Ensure box coordinates are within bounds
        y1 = max(0, min(y1, depth_map.shape[0] - 1))
        y2 = max(0, min(y2, depth_map.shape[0] - 1))
        x1 = max(0, min(x1, depth_map.shape[1] - 1))
        x2 = max(0, min(x2, depth_map.shape[1] - 1))

        # Check for invalid box dimensions
        if x2 <= x1 or y2 <= y1:
            return 100.0  # Default safe distance in meters

        # Extract depth values in the bounding box
        depth_values = depth_map[y1:y2, x1:x2].flatten()

        # Remove invalid values
        depth_values = depth_values[~np.isnan(depth_values)]
        depth_values = depth_values[depth_values > 0]

        if len(depth_values) == 0:
            return 100.0

        # Normalize and invert depth (lighter = closer)
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        normalized = (depth_values - depth_min) / (depth_max - depth_min + 1e-6)
        inverted = 1.0 - normalized

        # Use median inverted depth and scale to meters
        median_relative = np.median(inverted)
        depth_in_meters = median_relative * assumed_max_distance_m

        return depth_in_meters


class ObjectDetector:
    def __init__(self, model="yolov8n.pt"):
        """Initialize YOLO object detection model"""
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"ObjectDetector using device: {self.device}")
        
        # Load model
        self.model = YOLO(model)
        self.model.to(self.device)
    
    def detect(self, frame, conf=0.25):
        """Detect objects in a frame"""
        results = self.model(frame, conf=conf, verbose=False)[0]
        return results

class VLMProcessor:
    def __init__(self, model="llava-phi3"):
        """Initialize VLM for direction queries"""
        self.model = model
        print(f"VLMProcessor initialized with model: {model}")
    
    def get_direction(self, image, object_name, center_x, width):
        """Query VLM for directional advice"""
        # For efficiency, we'll use a direct prompt rather than open-ended query
        # Direction is determined by object position in frame
        if center_x < width/2:
            return "right"  # Object on left side, move right
        else:
            return "left"   # Object on right side, move left
    
    def query_llava(self, encoded_img, prompt):
        """Send image to LLaVa and get response"""
        payload = {
            "model": "llava-phi3",
            "prompt": prompt,
            "images": [encoded_img],
            "stream": False
        }

        try:
            res = requests.post("http://localhost:11434/api/generate", 
                               json=payload, timeout=30)  # Short timeout for responsiveness
            
            if res.ok:
                data = res.json()
                response = data.get("response", "").strip().lower()
                return response
            return ""
        except Exception as e:
            print(f"VLM query error: {e}")
            return ""
    
    def crop_and_encode(self, image, box, pad=20):
        """Crop object from image and encode as base64"""
        x1, y1, x2, y2 = map(int, map(round, box))
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(image.shape[1] - 1, x2 + pad)
        y2 = min(image.shape[0] - 1, y2 + pad)
        
        if x2 <= x1 or y2 <= y1:
            return None

        cropped = image[y1:y2, x1:x2]
        if cropped.size == 0:
            return None

        _, buffer = cv2.imencode('.jpg', cropped)
        return base64.b64encode(buffer).decode("utf-8")
    
def process_video(video_path, output_path=None):
    """Main video processing function"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_path}")
        return
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:  # NaN check
        fps = 30
    
    print(f"Video dimensions: {width}x{height}, FPS: {fps}")

    CONFIDENCE_THRESHOLD = 0.272 # DESCRIPTIVE MODE, your chosen threshold

    # CSV LOGGING
    csv_filename = os.path.join(OUTPUT_DIR, "descriptive_mode_results_vlm.csv")
    csv_file = open(csv_filename, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame",
        "object_name",
        "yolo_confidence",
        "depth_ft",
        "used_vlm",
        "output_text",
        "latency_ms",
        "direction"
    ])

    # Initialize detectors
    walkable_detector = WalkableAreaDetector(model_path="walkable_model1.pt")
    depth_estimator = DepthEstimator()
    object_detector = ObjectDetector()
    vlm_processor = VLMProcessor()

    # Video writer
    out = None
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width * 2, height)
        )

    # -------------------------
    # PROCESSING LOOP
    # -------------------------
    frame_count = 0

    try:
        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            display_frame = frame.copy()

            # Walkable areas
            walkable_areas = walkable_detector.detect(frame)
            display_frame = walkable_detector.visualize(display_frame, walkable_areas)

            # Depth
            depth_map, depth_colormap = depth_estimator.estimate_depth(frame)

            # YOLO Detection
            results = object_detector.detect(frame)

            # Defaults
            object_name = None
            confidence = None
            depth_ft = None
            used_vlm = 0
            vlm_text = ""
            direction = "none"

            # -------------------------------
            # OBJECT DETECTION LOOP
            # -------------------------------
            if hasattr(results, "boxes") and results.boxes is not None:
                for i, box in enumerate(results.boxes.xyxy):

                    # Convert YOLO tensor → numpy ints
                    if isinstance(box, torch.Tensor):
                        box = box.cpu().numpy()
                    x1, y1, x2, y2 = map(int, map(round, box))

                    confidence = results.boxes.conf[i].item()
                    class_id = int(results.boxes.cls[i].item())
                    object_name = results.names[class_id]

                    center_x = (x1 + x2) / 2

                    # Depth estimation
                    depth_m = depth_estimator.get_depth_at_box(depth_map, box)
                    depth_ft = depth_m * 3.28084
                    is_close = depth_ft < DISTANCE_THRESHOLD_FT

                    # RESET OUTPUTS
                    used_vlm = 0
                    vlm_text = ""
                    direction = "none"

                    # ==========================================
                    # CLEAN, CORRECT DECISION TREE
                    # ==========================================

                    # (1) Object too close → FAST MODE, no VLM
                    if depth_ft is not None and depth_ft < 10:
                        direction = "left" if center_x > width/2 else "right"
                        vlm_text = "[FAST MODE - TOO CLOSE]"
                        used_vlm = 0

                    # (2) YOLO is confident → FAST MODE
                    elif confidence >= CONFIDENCE_THRESHOLD:
                        direction = "left" if center_x > width/2 else "right"
                        vlm_text = "[FAST MODE - HIGH CONF]"
                        used_vlm = 0

                    # (3) YOLO uncertain → USE VLM
                    else:
                        encoded = vlm_processor.crop_and_encode(frame, box)
                        if encoded:
                            vlm_prompt = """
                            You are assisting a blind user. Describe the obstacle in this cropped image with detail, including:
                            1) What the object appears to be,
                            2) Its approximate position (left, center, right),
                            3) Any relevant physical characteristics (shape, size, color, risks),
                            4) A recommended navigation instruction (“move left”, “move right”, or “stop”),
                            5) A brief explanation of why.
                            Be descriptive and use complete sentences.
                            """

                            vlm_response = vlm_processor.query_llava(encoded, vlm_prompt)
                            vlm_text = vlm_response

                            if "left" in vlm_response.lower():
                                direction = "left"
                            elif "right" in vlm_response.lower():
                                direction = "right"
                            else:
                                direction = "none"

                            used_vlm = 1
                        else:
                            direction = "none"
                            vlm_text = "[CROP FAILED]"
                            used_vlm = 0

                    # --- Draw bounding boxes ---
                    color = (0, 0, 255) if is_close else (0, 255, 0)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

                    label = f"{object_name}: {depth_ft:.1f}ft | conf {confidence:.2f}"
                    cv2.putText(display_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Combine Depth + RGB
            depth_resized = cv2.resize(depth_colormap, (width, height))
            combined_view = np.hstack((display_frame, depth_resized))

            if out:
                out.write(combined_view)

            cv2.imshow("Navigation Assistant", combined_view)

            frame_count += 1
            latency_ms = (time.time() - start_time) * 1000

            # Log CSV entry
            csv_writer.writerow([
                frame_count,
                object_name,
                None if confidence is None else round(confidence, 3),
                None if depth_ft is None else round(depth_ft, 2),
                used_vlm,
                vlm_text,
                round(latency_ms, 2),
                direction
            ])

            if cv2.waitKey(1) == 27:
                break


    except KeyboardInterrupt:
        print("Processing stopped by user")

    finally:
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        csv_file.close()
        print("Processing complete")
        print(f"CSV results saved to {csv_filename}")




if __name__ == "__main__":
    import sys
    
    # Use command line argument or default video
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
         video_path = "/Users/rianachatterjee/Downloads/in-lab/YOLO_VLM_switch_off/videos/clip_28_to_29_ffmpeg.mp4"
        
    # Create output path
    video_name = os.path.basename(video_path).split('.')[0]
    output_path = os.path.join(OUTPUT_DIR, f"{video_name}_processed.mp4")
    
    # Process video
    process_video(video_path, output_path)
    
    print(f"Output saved to {output_path}")