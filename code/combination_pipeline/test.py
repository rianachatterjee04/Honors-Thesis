import base64, requests, json

# Encode your image
with open("/Users/rianachatterjee/Downloads/in-lab/vlm_test_frame.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "model": "MiniCPM-V:latest",
    "messages": [{"role": "user", "content": "What is in this picture?"}],
    "images": [img_b64],
    "stream": False
}

res = requests.post("http://localhost:11434/api/chat", json=payload)
print(json.dumps(res.json(), indent=2))
