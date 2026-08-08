import requests
import base64
import sys
import os
from ultralytics import YOLO
import cv2

def detect_closest_object_yolo(image_path):
    """Run YOLO on image and return the class of the closest detected object (based on bounding box size)."""
    model = YOLO("yolov8n.pt")  # Or replace with your fine-tuned model
    results = model(image_path)[0]

    if not results.boxes:
        return None

    # Choose the largest object based on bbox area
    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)
    names = results.names

    largest_area = 0
    closest_class = None

    for box, cls in zip(boxes, classes):
        x1, y1, x2, y2 = box
        area = (x2 - x1) * (y2 - y1)
        if area > largest_area:
            largest_area = area
            closest_class = names[cls]

    return closest_class

def test_ollama_image_query(image_path):
    """Send prompt and image to Ollama and replace [object] with YOLO result"""
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return

    print(f"[INFO] Using image: {image_path}")
    
    try:
        with open(image_path, "rb") as img_file:
            encoded_img = base64.b64encode(img_file.read()).decode("utf-8")
    except Exception as e:
        print(f"[ERROR] Failed to read or encode image: {e}")
        return

    prompt = "Is there an obstacle in this image within 6 feet? Reply with either 'clear' or '[object] ahead, move left/right'."

    payload = {
        "model": "llava-phi3",
        "prompt": prompt,
        "images": [encoded_img],
        "stream": False
    }

    print(f"[INFO] Sending request to Ollama with prompt:\n{prompt}")

    response_text = ""
    try:
        res = requests.post("http://localhost:11434/api/generate", json=payload, timeout=30)
        if res.ok:
            data = res.json()
            response_text = data.get("response", "")
        else:
            print(f"[WARN] /generate failed, trying /chat…")
            res = requests.post("http://localhost:11434/api/chat", json=payload, timeout=30)
            if res.ok:
                data = res.json()
                response_text = data.get("message", {}).get("content", "")
            else:
                print(f"[ERROR] Request failed: {res.text}")
                return
    except Exception as e:
        print(f"[ERROR] Failed to contact Ollama: {e}")
        return

    print(f"[INFO] Raw Ollama response:\n{response_text}")

    # Get closest object from YOLO
    closest_object = detect_closest_object_yolo(image_path)
    if closest_object:
        final_response = response_text.replace("[object]", closest_object)
        print(f"\n[🔁 Final Response]:\n{final_response}")
    else:
        print(f"\n[⚠️ YOLO] No objects detected. Original Ollama response kept.\n{response_text}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_ollama_image_query.py <image_path>")
        sys.exit(1)
        
    test_ollama_image_query(sys.argv[1])
