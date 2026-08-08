import os
import cv2
import base64
import requests
import time
from pathlib import Path
import json
import numpy as np
from ultralytics import YOLO
import torch

# Import MiDaS for depth estimation
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from torch.nn.functional import interpolate

# Configuration
OUTPUT_DIR = "output_dir"
MODEL_NAME = "llava-phi3:latest"
PROMPT_TEMPLATE = "Is there an obstacle in this image within 6 feet? Reply with either 'clear' or '[object] ahead, move left/right'."

# Ensure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_midas_model():
    """
    Load the MiDaS depth estimation model
    """
    # MiDaS v2.1 small model
    model_type = "MiDaS_small"
    
    # Download model if not available
    midas_path = "midas_model"
    os.makedirs(midas_path, exist_ok=True)
    model_path = os.path.join(midas_path, f"{model_type}.pt")
    
    if not os.path.exists(model_path):
        print(f"Downloading MiDaS model to {model_path}...")
        url = f"https://github.com/intel-isl/MiDaS/releases/download/v2_1/{model_type}.pt"
        os.system(f"curl -L -o {model_path} {url}")
    
    # Load model
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()
    
    # Define transforms for MiDaS input
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    if model_type == "MiDaS_small":
        transform = midas_transforms.small_transform
    else:
        transform = midas_transforms.dpt_transform
    
    return model, transform, device

def estimate_depth(model, transform, device, img):
    """
    Estimate depth using MiDaS model
    """
    # Transform input for MiDaS
    input_batch = transform(img).to(device)
    
    # Run inference
    with torch.no_grad():
        prediction = model(input_batch)
        
        # Resize to original resolution
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    # Convert to numpy array
    depth_map = prediction.cpu().numpy()
    
    # Normalize depth map to 0-1 range for visualization
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    normalized_depth = 255 * (depth_map - depth_min) / (depth_max - depth_min)
    normalized_depth = normalized_depth.astype("uint8")
    
    # Apply colormap for visualization
    depth_colormap = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_INFERNO)
    
    return depth_map, depth_colormap

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
    """Send a request to Ollama's LLaVa model"""
    payload = {
        "model": "llava-phi3",
        "prompt": prompt,
        "images": [encoded_img],
        "stream": False
    }

    try:
        print(f"→ Sending request to Ollama with prompt: '{prompt}'")
        res = requests.post("http://localhost:11434/api/generate", json=payload, timeout=20)
        print(f"← Response received. Status code: {res.status_code}")
        
        if res.ok:
            try:
                data = res.json()
                response_content = data.get("response", "").strip().lower()
                print(f"Extracted content: '{response_content}'")
                return response_content
            except Exception as e:
                print(f"⚠️ Error parsing response: {e}")
                return ""
        return ""
    except Exception as e:
        print(f"⚠️ Exception: {e}")
        return ""

def draw_danger_box(image, box, label):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(image, label, (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

def get_depth_at_box(depth_map, box):
    """
    Get average depth value within a bounding box
    """
    x1, y1, x2, y2 = map(int, map(round, box))
    
    # Ensure box coordinates are within depth map dimensions
    y1 = max(0, min(y1, depth_map.shape[0]-1))
    y2 = max(0, min(y2, depth_map.shape[0]-1))
    x1 = max(0, min(x1, depth_map.shape[1]-1))
    x2 = max(0, min(x2, depth_map.shape[1]-1))
    
    # Check if box has valid dimensions
    if x2 <= x1 or y2 <= y1:
        print(f"Warning: Invalid box dimensions: {x1},{y1},{x2},{y2}")
        return 5.0  # Return a default safe distance
    
    # Extract depth values in the bounding box
    box_depth = depth_map[y1:y2, x1:x2]
    
    if box_depth.size == 0:
        print("Warning: Empty box depth region")
        return 100.0  # Return a default safe distance
    
    # Get the median depth value to avoid outliers
    median_depth = np.median(box_depth)
    
    # Filter out extreme values (keep only middle 80% of values)
    depth_values = box_depth.flatten()
    depth_values = depth_values[~np.isnan(depth_values)]  # Remove NaN values
    if len(depth_values) > 10:  # Only filter if we have enough values
        q10 = np.percentile(depth_values, 10)
        q90 = np.percentile(depth_values, 90)
        filtered_values = depth_values[(depth_values >= q10) & (depth_values <= q90)]
        if len(filtered_values) > 0:
            median_depth = np.median(filtered_values)
    
    # Normalize to approximate meters (MiDaS produces relative depths)
    # This conversion is approximate and may need calibration
    # Scale factor is a hyperparameter that can be adjusted
    scale_factor = 0.03  # Reduced scale factor for more reasonable distances
    depth_in_meters = median_depth * scale_factor
    
    return depth_in_meters

def clean_and_format_response(response, object_name, distance_ft, center_x, width):
    """Ensure response strictly follows the required format for display"""
    # Strip any markdown code blocks
    if "```" in response:
        # Extract content between code blocks if possible
        code_parts = response.split("```")
        # Take the parts that aren't code block markers
        clean_parts = [part for i, part in enumerate(code_parts) if i % 2 == 1]
        if clean_parts:
            response = " ".join(clean_parts).strip()
        else:
            response = response.replace("```", "").strip()
    
    # Handle specific formatting
    if "[object]" in response and "ahead" in response and "move" in response:
        return response.replace("[object]", object_name)
    elif "move left" in response or "move right" in response:
        direction = "left" if "move left" in response else "right"
        return f"{object_name} ahead, move {direction}"
    elif "clear" in response.lower():
        return "clear"
    else:
        # For any other format, use depth-based decision with FIXED DIRECTION LOGIC
        # Move AWAY from the obstacle: if object is on left, move right; if on right, move left
        if distance_ft < 5.0:
            direction = "right" if center_x < width/2 else "left"
            return f"{object_name} ahead, move {direction}"
        else:
            return "clear"
        
def process_video(video_path=None, output_video=None):
    """Process video file or camera feed with YOLO and MiDaS."""
    if video_path is None:
        cap = cv2.VideoCapture(0)
    else:
        print(f"Opening video file: {video_path}")
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video source: {video_path}")
        return

    print(f"Video opened successfully: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    print(f"Video dimensions: {width}x{height}, FPS: {fps}")
    model = YOLO('yolov8n.pt')
    midas_model, midas_transform, device = load_midas_model()
    print(f"MiDaS model loaded. Using device: {device}")

    out, out_depth, out_combined = None, None, None
    if output_video:
        os.makedirs(os.path.dirname(output_video), exist_ok=True)
        out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        out_depth = cv2.VideoWriter(output_video.replace('.mp4', '_depth.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        out_combined = cv2.VideoWriter(output_video.replace('.mp4', '_combined.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width*2, height))

    frame_count = 0
    try:
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                if video_path: print("End of video file.")
                break

            result = model(frame)[0]
            depth_map, depth_colormap = estimate_depth(midas_model, midas_transform, device, frame)
            processed_frame = frame.copy()

            boxes = result.boxes.xyxy.cpu().numpy()
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, frame.shape[1])
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, frame.shape[0])
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            class_names = [result.names[class_id] for class_id in class_ids]

            # Initialize variables
            threat_detected = False
            collected_responses = []  # Collect all responses for numbered output

            # Process each detected object
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                object_name = class_names[i]
                
                # Get depth information
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                center_depth = depth_map[center_y, center_x]
                box_depth = get_depth_at_box(depth_map, box)
                depth_in_meters = box_depth
                distance_ft = depth_in_meters * 3.28084

                # Get cropped image for LLaVa analysis
                encoded_crop = crop_and_encode(frame, box)
                if encoded_crop is None:
                    continue

                # Try LLaVa first
                raw_response = query_llava_base64(encoded_crop, PROMPT_TEMPLATE)
                print(f"🧠 Object: {object_name}, distance: {distance_ft:.1f}ft, response: '{raw_response}'")
                
                # Always clean and format the response for strict adherence to template
                formatted_response = clean_and_format_response(
                    raw_response, object_name, distance_ft, center_x, width
                )
                
                # Show format correction in console
                if formatted_response != raw_response:
                    print(f"Format correction: '{raw_response}' → '{formatted_response}'")
                
                # Add to collected responses
                collected_responses.append(formatted_response)
                
                # Draw boxes based on formatted response only
                if "move" in formatted_response:
                    threat_detected = True
                    # Draw red box for obstacles
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(processed_frame, formatted_response, (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                else:
                    # Draw green box for clear paths
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(processed_frame, f"{formatted_response} | {distance_ft:.1f} ft", (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Print collected responses in numbered format
            if collected_responses:
                print("\n[🔁 Final Response]:")
                for i, response in enumerate(collected_responses, 1):
                    print(f"{i}. {response}")
                print("")  # Empty line for readability
            
            # Display the last response at the bottom of the screen
            if collected_responses:
                last_response = collected_responses[-1]
                color = (0, 0, 255) if "move" in last_response else (0, 255, 0)
                cv2.putText(processed_frame, last_response, (10, height - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Save frames periodically
            if frame_count % 30 == 0:
                frame_name = f"frame_{frame_count:04d}"
                cv2.imwrite(os.path.join(os.path.dirname(output_video), f"{frame_name}.jpg"), processed_frame)
                cv2.imwrite(os.path.join(os.path.dirname(output_video), f"{frame_name}_depth.jpg"), depth_colormap)
                combined = np.hstack((processed_frame, cv2.resize(depth_colormap, (processed_frame.shape[1], processed_frame.shape[0]))))
                cv2.imwrite(os.path.join(os.path.dirname(output_video), f"{frame_name}_combined.jpg"), combined)
                print(f"Saved snapshot and depth map for frame {frame_count}")

            # Prepare output videos
            depth_resized = cv2.resize(depth_colormap, (width, height))
            combined_view = np.hstack((processed_frame, depth_resized))
            if out: out.write(processed_frame)
            if out_depth: out_depth.write(depth_resized)
            if out_combined: out_combined.write(combined_view)

            # Display frames
            max_display_width = 1280
            if width > max_display_width/2:
                display_scale = max_display_width / (width * 2)
                display_size = (int(width*2*display_scale), int(height*display_scale))
                combined_display = cv2.resize(combined_view, display_size)
                cv2.imshow("Object Detection with Depth | Depth Map", combined_display)
            else:
                cv2.imshow("Object Detection with Depth", processed_frame)
                cv2.imshow("Depth Map", depth_colormap)

            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = 1.0 / elapsed_time if elapsed_time > 0 else 0
            print(f"Frame {frame_count} | FPS: {fps:.2f}")

            if cv2.waitKey(1) == 27:
                break

    except KeyboardInterrupt:
        print("Stopping video processing.")
    finally:
        cap.release()
        if out: out.release()
        if out_depth: out_depth.release()
        if out_combined: out_combined.release()
        cv2.destroyAllWindows()
        print(f"Video processing completed.")
        if output_video and os.path.exists(output_video):
            print(f"Main output: {output_video}")
            print(f"Depth output: {output_video.replace('.mp4', '_depth.mp4')}")
            print(f"Combined output: {output_video.replace('.mp4', '_combined.mp4')}")

if __name__ == "__main__":
    # Specify the path to the video file
    video_path = "YOLO_VLM_switch_off/videos/clip_107_to_159.mp4"
    
    # Create output directory based on video name
    video_name = os.path.basename(video_path).split('.')[0]
    output_dir = os.path.join(OUTPUT_DIR, video_name)
    os.makedirs(output_dir, exist_ok=True)
    
    output_video = os.path.join(output_dir, f'{video_name}_processed.mp4')
    process_video(video_path, output_video)
    print(f"Done. Processing complete for {video_path}")
    print(f"Output saved to {output_video}")