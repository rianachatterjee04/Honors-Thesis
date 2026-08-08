import os
import cv2
import time
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

def process_depth_stream(output_path):
    # Load YOLO model (adjust the model path as needed)
    model = YOLO('yolov8n.pt', device = "cuda")
    
    # Set up the RealSense pipeline to stream color and depth
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)   # Depth stream
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Color stream
    
    # Start streaming
    profile = pipeline.start(config)
    
    # Get depth scale to convert depth values to meters (typically ~0.001 for RealSense)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is:", depth_scale)
    
    # Create video writer for the output video file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_fps = 10  # Adjust this value as needed
    out = cv2.VideoWriter(output_path, fourcc, out_fps, (640, 480))
    
    frame_count = 0
    try:
        while True:
            start_time = time.time()
            
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Run YOLO detection on the color image
            results = model(color_image)[0]
            output_frame = color_image.copy()
            
            if results.boxes is not None:
                boxes = results.boxes.xyxy.cpu().numpy()  # format [x1, y1, x2, y2]
                classes = results.boxes.cls.cpu().numpy()
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    cls_id = int(classes[i])
                    class_name = model.names[cls_id]
                    
                    # Compute the center of the bounding box
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    
                    # Obtain the depth (in meters) at the center pixel
                    depth_in_meters = depth_image[center_y, center_x] * depth_scale
                    # Convert distance to feet (1 meter = 3.28084 feet)
                    distance_ft = depth_in_meters * 3.28084
                    
                    # Draw bounding box on the output frame
                    cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Create a text label with the class name and measured distance
                    label = f"{class_name}: {distance_ft:.1f} ft"
                    
                    # Determine text size and position (placing the label above the bounding box)
                    (text_width, text_height), baseline = cv2.getTextSize(label, 
                                                                          cv2.FONT_HERSHEY_SIMPLEX, 
                                                                          0.7, 2)
                    cv2.rectangle(output_frame, 
                                  (x1, y1 - text_height - baseline - 4), 
                                  (x1 + text_width, y1), 
                                  (255, 255, 255), -1)
                    cv2.putText(output_frame, label, (x1, y1 - 4), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Write the processed frame to the video file
            out.write(output_frame)
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")
            
            # Display the processed frame
            cv2.imshow("Depth Stream - Distance Estimation", output_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Optionally, print the instantaneous FPS
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                print("Current FPS:", 1.0 / elapsed_time)
    finally:
        pipeline.stop()
        out.release()
        cv2.destroyAllWindows()
        print(f"Done. Output saved to {output_path}")

if __name__ == "__main__":
    output_video = 'processed_depth_video.mp4'
    process_depth_stream(output_video)
