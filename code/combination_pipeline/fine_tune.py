import cv2
import os

# Path to video file
video_path = "YOLO_VLM_switch_off/videos/clip_20_to_107.mp4"
output_dir = "frames"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Open the video file
vidcap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not vidcap.isOpened():
    print("❌ Error: Could not open video. Check file path and format.")
    exit()

# Get video properties
fps = vidcap.get(cv2.CAP_PROP_FPS)  # Frames per second
total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) # Total frame count
duration = total_frames / fps # Video duration in seconds

print(f"Video FPS: {fps}")
print(f"Total Frames: {total_frames}")
print(f"Duration: {duration:.2f} seconds")

# Set frame extraction rate (1 frame per second)
frame_interval = int(fps)  # Extract every 'fps' frames to get 1 frame per second

count = 0 # Number of extracted frames
current_frame = 0

# Loop through frames
while vidcap.isOpened():
    success, image = vidcap.read()
    
    if not success:
        break # Stop if reading fails
    
    # Extract one frame per second
    if current_frame % frame_interval == 0:
        frame_name = os.path.join(output_dir, f"frame_{current_frame:04d}.jpg")
        cv2.imwrite(frame_name, image) # Save frame as JPEG
        print(f"✅ Saved: {frame_name}")
        count += 1
    
    current_frame += 1

vidcap.release() # Release video capture
print(f"✅ Extracted {count} frames (1 per second) from '{video_path}' to '{output_dir}'")