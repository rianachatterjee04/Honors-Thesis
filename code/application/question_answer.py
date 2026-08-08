import cv2
import numpy as np
import requests
import base64
import time
import csv
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt

# --------------------------
# Self-contained helper functions
# --------------------------

def capture_image_from_camera():
    """
    Capture a single image from the default webcam.
    
    Returns:
        A captured image as a NumPy array (BGR format) or None on error.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: Could not capture a frame from the camera.")
        return None
    return frame

def analyze_image_with_llama_vision(image, questions, model="llama3.2-vision"):
    """
    Analyze a captured image with Ollama's LLaVA model.
    
    Args:
        image: A NumPy array representing the captured image (BGR format).
        questions: A list of questions to ask about the image.
        model: The Ollama model name to use (default: "llama3.2-vision").
    
    Returns:
        A list of dictionaries containing each question, its answer, and the response time.
    """
    # Convert image from BGR to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Display the captured image using matplotlib
    plt.figure(figsize=(8, 8))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title('Captured Image')
    plt.show()
    
    # Convert the captured image to JPEG bytes in memory and encode as base64
    retval, buffer = cv2.imencode('.jpg', image)
    if not retval:
        print("Error: Could not encode image.")
        return []
    base64_image = base64.b64encode(buffer).decode('utf-8')
    
    results = []
    for question in questions:
        start_time = time.time()
        # Prepare the API request payload
        data = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": question,
                "images": [base64_image]
            }],
            "stream": False
        }
        try:
            response = requests.post("http://localhost:11434/api/chat", json=data)
            if response.status_code == 200:
                response_data = response.json()
                answer = response_data.get("message", {}).get("content", "No response received")
                elapsed_time = time.time() - start_time
                print(f"Q: {question}")
                print(f"A: {answer}")
                print(f"Time: {elapsed_time:.2f}s\n")
                results.append({
                    "question": question,
                    "answer": answer,
                    "time": elapsed_time
                })
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
                results.append({
                    "question": question,
                    "answer": f"Error: {response.status_code}",
                    "time": 0
                })
        except Exception as e:
            print(f"Request failed: {e}")
            results.append({
                "question": question,
                "answer": f"Exception: {str(e)}",
                "time": 0
            })
    return results

def save_results(results, filename=None):
    """
    Save analysis results to a CSV file.
    
    Args:
        results: List of dictionaries containing questions, answers, and timings.
        filename: Optional filename; if None, a timestamped filename is generated.
    
    Returns:
        The filename of the saved CSV.
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"image_analysis_{timestamp}.csv"
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["question", "answer", "time"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {filename}")
    return filename

# --------------------------
# Default questions to ask
# --------------------------

DEFAULT_QUESTIONS = [
    "Describe the terrain and surface in this image.",
    "What potential hazards or obstacles can be seen?",
    "Are there any notable environmental conditions?",
    "What safety considerations should be taken into account?"
]

# --------------------------
# Main function for analysis using camera input
# --------------------------

def run_ollama_analysis(model="llama3.2-vision", custom_questions=None):
    questions = custom_questions if custom_questions else DEFAULT_QUESTIONS
    print("Capturing image from camera...")
    image = capture_image_from_camera()
    if image is None:
        print("No image captured.")
        return
    results = analyze_image_with_llama_vision(image, questions, model)
    save_results(results)
    if results:
        avg_time = sum(r["time"] for r in results) / len(results)
    else:
        avg_time = 0
    print(f"\nAnalysis completed with average response time: {avg_time:.2f}s")
    
if __name__ == "__main__":
    run_ollama_analysis()
