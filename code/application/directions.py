import os
import cv2
import time
import numpy as np
from ultralytics import YOLO

# ------------------------
# Self-contained helper functions
# ------------------------

def relative_distance_to_feet(rel_dist):
    """
    Convert a relative distance (0 to 1) to an approximate distance in feet.
    For example, assume that a relative distance of 1 (i.e. at the bottom of the frame)
    corresponds to 7 ft away.
    """
    max_feet = 7.0
    return rel_dist * max_feet

def choose_vlm_model(dist_feet):
    """
    Dummy function to select a Visual Language Model (VLM) based on the distance.
    For example, if the distance is less than or equal to 7 ft, we return a model name;
    otherwise, we return None.
    """
    if dist_feet <= 7.0:
        return "moondream"
    return None

def query_ollama(model_name, prompt):
    """
    Dummy function to simulate querying a Visual Language Model (like Ollama).
    Instead of making an API call, it returns a hard-coded directional instruction.
    In a real system, you would replace this with a call to an external API.
    """
    # Example behavior based on prompt content; this is just a placeholder.
    if "turn" in prompt.lower():
        return "Turn left."
    # Default instruction.
    return "Move forward."

# ------------------------
# ProximityDetector Class
# ------------------------

class ProximityDetector:
    """
    Integrates object detection (using YOLO-v8), distance-based VLM switching,
    and directional instructions.
    """
    def __init__(self, model_path='yolov8n.pt'):
        # Directly create the YOLO object detector.
        self.model = YOLO(model_path)
        # Define the immediate proximity zone:
        # Bottom 40% of the frame, centered horizontally (40% of frame width).
        self.proximity_height = 0.4
        self.proximity_width = 0.4

    def detect_objects(self, frame):
        """
        Run YOLO detection on the frame.
        Returns bounding boxes, class indices, and class names.
        """
        results = self.model(frame)[0]
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()  # Format: [x1, y1, x2, y2]
            class_indices = results.boxes.cls.cpu().numpy()
            class_names = [self.model.names[int(idx)] for idx in class_indices]
            return boxes, class_indices, class_names
        else:
            return np.empty((0, 4)), np.array([]), []

    def is_in_proximity(self, box, frame_shape):
        """
        Check if an object's bounding box is within the defined bottom-center region.
        Returns (True, relative_distance) if the object is in the zone, else (False, 0.0).
        The relative distance is computed as a number between 0 (top of zone) and 1 (bottom of frame).
        """
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = box
        box_center_x = (x1 + x2) / 2
        box_bottom = y2

        # Define the bottom-center zone.
        zone_left = width * (0.5 - self.proximity_width / 2)
        zone_right = width * (0.5 + self.proximity_width / 2)
        zone_top = height * (1 - self.proximity_height)

        if (zone_left < box_center_x < zone_right) and (box_bottom > zone_top):
            # Compute a relative distance within the zone.
            rel_dist = (box_bottom - zone_top) / (height - zone_top)
            return True, rel_dist
        return False, 0.0

    def process_frame(self, frame):
        """
        Processes a frame by:
          1. Detecting objects using YOLO-v8.
          2. For each object classified as 'person', 'bicycle', or 'car' in the bottom-center zone:
             - Computes relative distance.
             - Converts that to an approximate distance in feet.
             - Chooses a VLM model and queries it for a direction.
             - Overlays a red bounding box on the frame.
          3. Returns the processed frame and a list of directional instructions.
        """
        processed = frame.copy()
        boxes, class_indices, class_names = self.detect_objects(frame)
        instructions = []

        for box, cls_name in zip(boxes, class_names):
            if cls_name in ['person', 'bicycle', 'car']:
                in_prox, rel_dist = self.is_in_proximity(box, frame.shape)
                if in_prox:
                    dist_feet = relative_distance_to_feet(rel_dist)
                    model_name = choose_vlm_model(dist_feet)
                    if model_name is not None:
                        prompt = (
                            f"An obstacle (a {cls_name}) is about {dist_feet:.1f} ft away. "
                            "We want to guide the user to proceed safely. "
                            "Suggest a short direction (turn left/right, move forward, stop)."
                        )
                        response = query_ollama(model_name, prompt)
                        instructions.append(response)
                    # Draw bounding box around the obstacle.
                    cv2.rectangle(
                        processed,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        (0, 0, 255), 2  # Red color.
                    )
        return processed, instructions

# ------------------------
# Video Processing Function
# ------------------------

def process_video_from_webcam(output_path):
    """
    Connects to the webcam, processes a continuous video stream using the ProximityDetector,
    overlays directional instructions on each frame, and writes the output video to file.
    """
    os.makedirs("videos", exist_ok=True)
    detector = ProximityDetector()

    # Open the default webcam (device 0). Adjust if necessary.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set desired resolution.
    width = 640
    height = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Create a VideoWriter to save the processed video.
    out_fps = 10  # Adjust the FPS as needed.
    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          out_fps,
                          (width, height))

    frame_count = 0
    print("Starting video capture; press 'q' to quit.")
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, instructions = detector.process_frame(frame)

        # Overlay each directional instruction on the frame.
        for i, instruction in enumerate(instructions):
            cv2.putText(processed_frame,
                        instruction,
                        (10, 30 + (i * 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2)

        out.write(processed_frame)
        cv2.imshow("Directional Proximity Stream", processed_frame)
        frame_count += 1

        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: Instructions: {instructions}")

        # Exit loop if 'q' is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        elapsed_time = time.time() - start_time
        if elapsed_time > 0:
            print("Current FPS:", 1.0 / elapsed_time)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Done. Output saved to {output_path}")

# ------------------------
# Main Execution
# ------------------------

if __name__ == "__main__":
    output_video = 'processed_directions_video.mp4'
    process_video_from_webcam(output_video)
