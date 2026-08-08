import base64
import io
import requests
import time
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Define detail prompt for level "detailed"
DETAILED_PROMPT = "Provide a comprehensive description of this image, including all visible elements, spatial relationships, and any text."

def capture_image_from_camera(camera_index=0):
    """
    Capture a single frame from the webcam.
    
    Args:
        camera_index: Index of the camera device (default 0).
    
    Returns:
        The captured image in RGB color space as a NumPy array, or None if unsuccessful.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return None

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: Failed to capture image from camera.")
        return None

    # Convert the captured BGR frame (OpenCV default) to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb

def describe_captured_image(image_array, prompt=DETAILED_PROMPT):
    """
    Describe a captured image (provided as a NumPy array) using the Ollama moondream model.
    
    Args:
        image_array: The captured image (RGB, as a NumPy array).
        prompt: The text prompt for description.
    
    Returns:
        The description text returned from Ollama.
    """
    # Display the captured image.
    image_pil = Image.fromarray(image_array)
    plt.figure(figsize=(8, 8))
    plt.imshow(image_pil)
    plt.axis('off')
    plt.title('Captured Image')
    plt.show()

    # Convert the PIL image to a base64-encoded JPEG.
    buffered = io.BytesIO()
    image_pil.save(buffered, format="JPEG")
    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    start_time = time.time()

    # Prepare the API request payload.
    data = {
        "model": "moondream",
        "prompt": prompt,
        "images": [base64_image],
        "stream": False
    }

    try:
        response = requests.post("http://localhost:11434/api/generate", json=data)
        if response.status_code == 200:
            description = response.json().get("response", "No description generated")
            elapsed_time = time.time() - start_time
            print("\nDetailed Description:")
            print("=" * 80)
            print(description)
            print("=" * 80)
            print(f"Processing time: {elapsed_time:.2f} seconds")
            return description
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return f"Error: {response.status_code}"
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to Ollama. Make sure Ollama is running with 'ollama serve'.")
        print("If you haven't installed the model, run: ollama pull moondream")
        return "Connection error"

def capture_and_describe():
    """Capture an image from the camera and describe it using the detailed prompt."""
    # Capture an image from the default camera.
    image = capture_image_from_camera(0)
    if image is None:
        print("Failed to capture image.")
        return

    # Use the detailed prompt (prompt level 3)
    describe_captured_image(image, prompt=DETAILED_PROMPT)

if __name__ == "__main__":
    capture_and_describe()
