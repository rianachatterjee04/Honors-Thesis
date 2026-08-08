import os
import cv2
import time
import torch
import numpy as np
import requests
from ultralytics import YOLO

# Configuration constants
OUTPUT_DIR = "output_dir"
DISPLAY_SCALE = 1.0  # Scale display for high-res screens
SAVE_FRAMES = True  # Save frames for debugging/demo
VLM_ENABLED = True  # Enable VLM direction queries
WARNING_THRESHOLD_AREA = 0.15  # Objects larger than this fraction of the screen are considered close

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

class WalkableAreaDetector:
    def __init__(self, model_path="walkable_model.pt"):
        """Initialize walkable area detection model"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"WalkableAreaDetector using device: {self.device}")
        
        # Load the model
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            print(f"Loaded walkable area model from {model_path}")
        except Exception as e:
            print(f"Error loading walkable model: {e}")
            self.model = None
    
    def detect(self, frame, conf=0.25):
        """Detect walkable areas in a frame"""
        if self.model is None:
            return []
        
        try:
            results = self.model(frame, conf=conf, verbose=False)
            walkable_areas = []
            
            for result in results:
                if hasattr(result, 'masks') and result.masks is not None:
                    for mask in result.masks.xy:
                        if len(mask) >= 3:  # Need at least 3 points for a polygon
                            walkable_areas.append(mask)
            
            return walkable_areas
        except Exception as e:
            print(f"Error in walkable area detection: {e}")
            return []
            
    def visualize(self, frame, walkable_areas):
        """Render walkable areas on the frame"""
        if not walkable_areas:
            return frame
            
        # Create a copy of the frame
        output = frame.copy()
        
        # Create a mask for walkable areas
        mask = np.zeros_like(frame)
        
        # Fill polygons with green
        for area in walkable_areas:
            area_points = np.array(area, dtype=np.int32)
            cv2.fillPoly(mask, [area_points], (0, 255, 0))
            
        # Apply the mask with alpha blending (0.3 alpha for the green overlay)
        output = cv2.addWeighted(output, 0.7, mask, 0.3, 0)
        
        # Add outlines for better visibility
        for area in walkable_areas:
            area_points = np.array(area, dtype=np.int32)
            cv2.polylines(output, [area_points], True, (0, 200, 0), 2)
            
        return output

class ProximityEstimator:
    """Replacement for DepthEstimator that uses object size as a proxy for distance"""
    
    def __init__(self, frame_width, frame_height):
        """Initialize proximity estimator with frame dimensions"""
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_area = frame_width * frame_height
        print(f"ProximityEstimator initialized with frame size: {frame_width}x{frame_height}")
    
    def estimate_proximity(self, box):
        """Estimate if an object is close based on its size relative to the frame"""
        # Get box dimensions
        x1, y1, x2, y2 = map(int, map(round, box))
        
        # Calculate box area
        box_width = x2 - x1
        box_height = y2 - y1
        box_area = box_width * box_height
        
        # Calculate ratio of box area to frame area
        area_ratio = box_area / self.frame_area
        
        # Calculate approximate "pseudo-distance" - not actual distance but a proxy
        # Higher values = further away (inverse relationship with area)
        pseudo_distance_ft = 10.0 / (area_ratio * 20) if area_ratio > 0 else 100.0
        
        # Limit the range to reasonable values (2ft - 50ft)
        pseudo_distance_ft = max(2.0, min(50.0, pseudo_distance_ft))
        
        return pseudo_distance_ft, area_ratio > WARNING_THRESHOLD_AREA

class ObjectDetector:
    def __init__(self, model="yolov8n.pt"):
        """Initialize YOLO object detection model"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

def process_video(video_path, output_path=None):
    """Main video processing function"""
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_path}")
        return
        
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    
    print(f"Video dimensions: {width}x{height}, FPS: {fps}")
    
    # Initialize detectors
    walkable_detector = WalkableAreaDetector()
    proximity_estimator = ProximityEstimator(width, height)
    object_detector = ObjectDetector()
    vlm_processor = VLMProcessor()
    
    # Initialize video writer
    out = None
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        out = cv2.VideoWriter(output_path, 
                             cv2.VideoWriter_fourcc(*'mp4v'), 
                             fps, (width, height))
    
    # Process frames
    frame_count = 0
    try:
        while True:
            start_time = time.time()
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Make a copy for visualization
            display_frame = frame.copy()
            
            # Step 1: Detect walkable areas
            walkable_areas = walkable_detector.detect(frame)
            display_frame = walkable_detector.visualize(display_frame, walkable_areas)
            
            # Step 2: Detect objects
            results = object_detector.detect(frame)
            
            # Step 3: Process each detected object
            warnings = []
            
            if hasattr(results, 'boxes') and results.boxes is not None:
                for i, box in enumerate(results.boxes.xyxy):
                    # Extract box coordinates
                    x1, y1, x2, y2 = map(int, box[:4])
                    confidence = results.boxes.conf[i].item()
                    class_id = int(results.boxes.cls[i].item())
                    object_name = results.names[class_id]
                    
                    # Get object center
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    # Estimate proximity based on object size
                    pseudo_distance_ft, is_close = proximity_estimator.estimate_proximity(box)
                    
                    # Get movement direction if object is close
                    direction = "unknown"
                    if is_close and VLM_ENABLED:
                        # For demo purposes, just use position-based direction
                        direction = vlm_processor.get_direction(frame, object_name, center_x, width)
                        
                        # Create warning message
                        warning = f"{object_name} detected, move {direction}"
                        warnings.append(warning)
                    
                    # Draw bounding box
                    color = (0, 0, 255) if is_close else (0, 255, 0)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label with estimated distance
                    label = f"{object_name}: ~{pseudo_distance_ft:.1f}ft"
                    cv2.putText(display_frame, label, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Step 4: Display warnings
            if warnings:
                # Add warning text to display
                warning_text = " | ".join(warnings)
                cv2.putText(display_frame, warning_text, (10, height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(f"Warnings: {', '.join(warnings)}")
            
            # Save frame if needed
            if SAVE_FRAMES and frame_count % 30 == 0:
                frame_path = os.path.join(OUTPUT_DIR, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(frame_path, display_frame)
            
            # Write to output video
            if out:
                out.write(display_frame)
            
            # Display frame
            # Resize for display if needed
            if DISPLAY_SCALE != 1.0:
                display_w = int(width * DISPLAY_SCALE)
                display_h = int(height * DISPLAY_SCALE)
                display_frame = cv2.resize(display_frame, (display_w, display_h))
            
            cv2.imshow("Navigation Assistant", display_frame)
            
            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - start_time
            fps_current = 1.0 / elapsed if elapsed > 0 else 0
            print(f"Frame {frame_count} | FPS: {fps_current:.2f}")
            
            # Exit on ESC
            if cv2.waitKey(1) == 27:
                break
                
    except KeyboardInterrupt:
        print("Processing stopped by user")
    finally:
        # Clean up
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        print("Processing complete")

if __name__ == "__main__":
    import sys
    
    # Use command line argument or default video
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
         video_path = "YOLO_VLM_switch_off/videos/clip_20_to_107.mp4"
        
    # Create output path
    video_name = os.path.basename(video_path).split('.')[0]
    output_path = os.path.join(OUTPUT_DIR, f"{video_name}_processed_no_midas.mp4")
    
    # Process video
    process_video(video_path, output_path)
    
    print(f"Output saved to {output_path}")