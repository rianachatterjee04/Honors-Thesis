from flask import Flask, request, jsonify
import base64
import io
from PIL import Image
import requests

app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "granite3.2-vision:2b"

@app.post("/vlm")
def vlm():
    data = request.json
    image_b64 = data.get("image", "")
    prompt = data.get("prompt", "")

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    }

    # Only send images if not empty
    if image_b64:
        payload["images"] = [image_b64]

    try:
        r = requests.post(OLLAMA_URL, json=payload)

        # DEBUG — see exactly what Ollama returns
        print("RAW OLLAMA JSON:", r.json())

        resp = r.json()

        text = (
            resp.get("response") or
            resp.get("output") or
            resp.get("message", {}).get("content") or
            ""
        )

        return jsonify({"response": text})

    except Exception as e:
        print("VLM ERROR:", e)
        return jsonify({"response": ""})


if __name__ == "__main__":
    print("VLM server running on port 5000")
    app.run(host="0.0.0.0", port=5000)

