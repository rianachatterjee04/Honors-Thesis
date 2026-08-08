import os
import cv2
import time
import numpy as np
from ultralytics import YOLO

class ProximityDetector:
    def __init__(self):
        # Load models
        self.walkable_model = YOLO('/home/unitree/genassist_poc-main/walkable_model.pt')  # For walkable areas
        self.object_model = YOLO('yolov8n-seg.pt')

        # Define immediate proximity zone (area right in front)
        self.proximity_height = 0.4  # Bottom 40% of frame
        self.proximity_width = 0.4   # Center 40% of frame width

    def draw_mask_safely(self, frame, mask, color, alpha=0.5):
        """Safely draw a mask on a frame with validation"""
        try:
            if mask is None or len(mask) < 3:  
                return frame

            mask_points = np.array(mask, dtype=np.int32)
            if mask_points.shape[0] < 3:
                return frame

            overlay = frame.copy()
            cv2.fillPoly(overlay, [mask_points], color)
            return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

        except Exception as e:
            print(f"Warning: Could not draw mask: {e}")
            return frame

    def is_in_proximity(self, box, frame_shape):
        """Check if object is very close to the robot's path"""
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = box
        box_center_x = (x1 + x2) / 2
        box_bottom = y2

        # Define immediate front zone
        zone_left = width * (0.5 - self.proximity_width / 2)
        zone_right = width * (0.5 + self.proximity_width / 2)
        zone_top = height * (1 - self.proximity_height)

        # Check if object is in immediate front zone
        if (zone_left < box_center_x < zone_right) and (box_bottom > zone_top):
            relative_distance = (box_bottom - zone_top) / (height - zone_top)
            return True, relative_distance
        return False, 0

    def process_frame(self, frame):
        height, width = frame.shape[:2]
        processed = frame.copy()

        # 1. Detect walkable zones
        walkable_results = self.walkable_model.predict(frame, show=False)
        if walkable_results[0].masks is not None:
            for mask in walkable_results[0].masks.xy:
                processed = self.draw_mask_safely(
                    processed,
                    mask,
                    color=(0, 255, 0),  # Green
                    alpha=0.3
                )
                if len(mask) >= 3:
                    cv2.polylines(processed, [np.int32(mask)], True, (0, 255, 0), 2)

        # 2. Detect and track objects
        object_results = self.object_model.track(frame, persist=True)[0]

        # Track the closest obstacle
        closest_distance = 0
        has_close_obstacle = False

        if object_results.boxes is not None and object_results.masks is not None:
            boxes = object_results.boxes.xyxy.cpu().numpy()
            classes = object_results.boxes.cls.cpu().numpy()
            masks = object_results.masks.xy
            track_ids = (object_results.boxes.id.cpu().numpy()
                         if object_results.boxes.id is not None else None)

            for i, (box, cls, mask) in enumerate(zip(boxes, classes, masks)):
                processed = self.draw_mask_safely(
                    processed,
                    mask,
                    color=(255, 165, 0),  # Orange
                    alpha=0.5
                )

                # Add tracking ID + label
                if track_ids is not None and len(mask) > 0:
                    track_id = int(track_ids[i])
                    class_name = self.object_model.names[int(cls)]
                    label = f"ID {track_id} - {class_name}"
                    x_min, y_min = mask.min(axis=0)
                    cv2.putText(processed, label,
                                (int(x_min), int(y_min) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 255, 255), 2)

                # Check proximity for obstacles (person, bicycle, car)
                class_name = self.object_model.names[int(cls)]
                if class_name in ['person', 'bicycle', 'car']:
                    in_proximity, distance = self.is_in_proximity(box, frame.shape)
                    if in_proximity and distance > 0.3:
                        has_close_obstacle = True
                        if distance > closest_distance:
                            closest_distance = distance
                        cv2.rectangle(
                            processed,
                            (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])),
                            (0, 0, 255), 2
                        )

            # Add warning text if obstacle is detected
            if has_close_obstacle:
                warning_text = "Obstacle ahead, please move"
                text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX,
                                            1.2, 3)[0]
                text_x = (width - text_size[0]) // 2
                cv2.putText(processed, warning_text,
                            (text_x, height - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (0, 0, 255), 3)

        return processed


def process_video_from_webcam(output_path):
    """
    Capture frames from the webcam and process them at whatever speed 
    your pipeline (model inference) can handle. Then, write them to 
    a video file at that *same* average FPS, so the playback speed
    is consistent with your actual processing speed.
    """
    os.makedirs("videos", exist_ok=True)
    detector = ProximityDetector()

    cap = cv2.VideoCapture(4)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    width = 640  # Set width to 640 for 480p resolution
    height = 480  # Set height to 480 for 480p resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # We won't rely on camera's FPS. Instead, we'll measure the time we
    # spend on each frame and maintain an "average processing FPS".
    # If your pipeline runs at ~8-10 FPS, let's pick 10 as an example.
    # Or you can dynamically update it based on the actual processing speed.
    out_fps = 1

    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          out_fps,  # We'll write the file at ~10 FPS
                          (width, height))

    frame_count = 0
    while True:
        start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = detector.process_frame(frame)
        out.write(processed_frame)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed frame {frame_count}")

        # Measure how long it took to process this frame
        elapsed_time = time.time() - start_time
        # You can print it out to see your actual FPS
        print("Current FPS:", 1.0 / elapsed_time if elapsed_time > 0 else 0)

    cap.release()
    out.release()


if __name__ == "__main__":
    output_video = 'processed_video_from_webcam_480p.mp4'
    process_video_from_webcam(output_video)
    print(f"Done. Output saved to {output_video}")

