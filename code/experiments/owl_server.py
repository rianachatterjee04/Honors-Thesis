from flask import Flask, request, jsonify
import base64
import io
from PIL import Image
import requests as http_requests
import torch
import numpy as np

app = Flask(__name__)

# ── Ollama config (Granite VLM description endpoint) ──────────────────────────
OLLAMA_URL    = "http://localhost:11434/api/generate"
GRANITE_MODEL = "granite3.2-vision:2b"

# ── OWLv2 — class-agnostic open-vocabulary object detector ────────────────────
# Detects anything using broad natural-language queries — no class list needed.
# Model is ~330MB and runs on MPS (Apple Silicon).
print("[OWLv2] Loading model...")
from transformers import Owlv2Processor, Owlv2ForObjectDetection

OWL_MODEL_ID = "google/owlv2-base-patch16-ensemble"
owl_processor = Owlv2Processor.from_pretrained(OWL_MODEL_ID)
owl_model     = Owlv2ForObjectDetection.from_pretrained(OWL_MODEL_ID)

device = "mps" if torch.backends.mps.is_available() else "cpu"
owl_model = owl_model.to(device)
owl_model.eval()
print(f"[OWLv2] Ready on {device}.")

# Broad queries — OWLv2 will box anything that matches, no specific classes needed
AGNOSTIC_QUERIES = [["an object", "a person", "a thing", "furniture", "a surface"]]

# Confidence threshold for OWLv2 detections — lower = more boxes, higher = fewer
OWL_CONF_THRESHOLD = 0.12
OWL_NMS_THRESHOLD  = 0.3


def b64_to_pil(b64_str: str) -> Image.Image:
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def nms(boxes, scores, iou_threshold=0.3):
    """Simple NMS to remove duplicate overlapping boxes."""
    if len(boxes) == 0:
        return []
    boxes  = np.array(boxes)
    scores = np.array(scores)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


# ── /vlm — Granite semantic description via Ollama ────────────────────────────
@app.post("/vlm")
def vlm():
    data      = request.json
    image_b64 = data.get("image", "")
    prompt    = data.get("prompt", "")
    payload   = {"model": GRANITE_MODEL, "prompt": prompt, "stream": False}
    if image_b64:
        payload["images"] = [image_b64]
    try:
        r    = http_requests.post(OLLAMA_URL, json=payload, timeout=60.0)
        resp = r.json()
        text = (
            resp.get("response") or
            resp.get("output") or
            resp.get("message", {}).get("content") or ""
        )
        return jsonify({"response": text})
    except Exception as e:
        print(f"[VLM] Error: {e}")
        return jsonify({"response": ""})


# ── /detect — OWLv2 class-agnostic detection ──────────────────────────────────
@app.post("/detect")
def detect():
    """
    Request:  { "image": "<base64 jpg>" }
              (no 'objects' list needed — OWLv2 finds anything)

    Response: { "detections": [
                  {"label": "an object", "score": 0.43,
                   "x_min": 0.1, "y_min": 0.2, "x_max": 0.4, "y_max": 0.8},
                  ...
                ],
                "pseudo_conf": 0.67   <- ready-to-use BO confidence score
              }
    Coordinates are normalized 0–1.
    pseudo_conf: scalar 0–1 summarising how much the OWLv2 sees in this frame.
                 High = scene well-covered, Low = uncertain → trigger Granite VLM.
    """
    data      = request.json
    image_b64 = data.get("image", "")

    if not image_b64:
        return jsonify({"error": "No image provided"}), 400

    try:
        pil_img = b64_to_pil(image_b64)
    except Exception as e:
        return jsonify({"error": f"Could not decode image: {e}"}), 400

    w, h = pil_img.size

    try:
        inputs = owl_processor(
            text=AGNOSTIC_QUERIES,
            images=pil_img,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = owl_model(**inputs)

        target_sizes = torch.tensor([[h, w]], dtype=torch.float32)
        results = owl_processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=OWL_CONF_THRESHOLD
        )[0]

        boxes_px = results["boxes"].cpu().numpy()   # (N, 4) in pixel coords
        scores   = results["scores"].cpu().numpy()  # (N,)
        labels   = results["labels"].cpu().numpy()  # (N,) — index into AGNOSTIC_QUERIES[0]

        # Convert to normalized coords
        norm_boxes = []
        for box in boxes_px:
            x1 = float(np.clip(box[0] / w, 0.0, 1.0))
            y1 = float(np.clip(box[1] / h, 0.0, 1.0))
            x2 = float(np.clip(box[2] / w, 0.0, 1.0))
            y2 = float(np.clip(box[3] / h, 0.0, 1.0))
            norm_boxes.append([x1, y1, x2, y2])

        # NMS on normalized coords
        keep = nms(norm_boxes, scores.tolist(), OWL_NMS_THRESHOLD) if norm_boxes else []

        detections = []
        for idx in keep:
            box   = norm_boxes[idx]
            score = float(scores[idx])
            label_idx = int(labels[idx])
            label_str = AGNOSTIC_QUERIES[0][label_idx] if label_idx < len(AGNOSTIC_QUERIES[0]) else "object"
            detections.append({
                "label": label_str,
                "score": round(score, 4),
                "x_min": round(box[0], 4),
                "y_min": round(box[1], 4),
                "x_max": round(box[2], 4),
                "y_max": round(box[3], 4),
            })

        # pseudo_conf: weighted by both count and average confidence score
        # High pseudo_conf = OWLv2 sees stuff confidently → maybe skip Granite
        # Low pseudo_conf = uncertain scene → trigger Granite description
        MAX_EXPECTED = 8
        if detections:
            avg_score  = float(np.mean([d["score"] for d in detections]))
            count_norm = min(len(detections) / MAX_EXPECTED, 1.0)
            pseudo_conf = float(np.clip((avg_score + count_norm) / 2.0, 0.0, 1.0))
        else:
            pseudo_conf = 0.0

        print(f"[OWLv2] {len(detections)} detections | pseudo_conf={pseudo_conf:.3f}")
        return jsonify({"detections": detections, "pseudo_conf": pseudo_conf})

    except Exception as e:
        print(f"[OWLv2] Detection error: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"detections": [], "pseudo_conf": 0.0})


if __name__ == "__main__":
    print("VLM server running on port 5000")
    app.run(host="0.0.0.0", port=5000)