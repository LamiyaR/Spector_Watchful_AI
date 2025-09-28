
import io
import os
from pathlib import Path
from typing import Tuple

from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import cv2

try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None

APP_ROOT = Path(__file__).parent
MODELS_DIR = APP_ROOT / "models"

DEFAULT_YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "best.pt")
DEFAULT_HAAR_XML = os.environ.get("HAAR_XML", "HC-classifier.xml")

app = Flask(__name__)

_yolo_model = None
_haar_cascade = None

def load_yolo(weights_name: str = DEFAULT_YOLO_WEIGHTS):
    global _yolo_model
    if _yolo_model is None:
        weights_path = (MODELS_DIR / weights_name)
        if not weights_path.exists():
            candidates = list(MODELS_DIR.glob("*.pt"))
            if not candidates:
                raise FileNotFoundError("No YOLO .pt weights found in ./models. Put your weights file there.")
            weights_path = candidates[0]
        if YOLO is None:
            raise RuntimeError("ultralytics not installed. Install it from requirements.txt.")
        _yolo_model = YOLO(str(weights_path))
    return _yolo_model

def load_haar(xml_name: str = DEFAULT_HAAR_XML):
    global _haar_cascade
    if _haar_cascade is None:
        xml_path = (MODELS_DIR / xml_name)
        if not xml_path.exists():
            candidates = list(MODELS_DIR.glob("*.xml"))
            if not candidates:
                raise FileNotFoundError("No Haar Cascade .xml found in ./models.")
            xml_path = candidates[0]
        _haar_cascade = cv2.CascadeClassifier(str(xml_path))
    return _haar_cascade

def read_image(file_storage) -> np.ndarray:
    data = np.frombuffer(file_storage.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img

def annotate_yolo(img: np.ndarray):
    model = load_yolo()
    results = model(img)[0]
    counts = {}
    annotated = img.copy()
    if hasattr(results, "boxes") and results.boxes is not None:
        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu().numpy()) if hasattr(box, "conf") else 0.0
            cls_id = int(box.cls[0].cpu().numpy()) if hasattr(box, "cls") else -1
            label = results.names.get(cls_id, str(cls_id)) if hasattr(results, "names") else str(cls_id)
            counts[label] = counts.get(label, 0) + 1
            x1, y1, x2, y2 = xyxy.tolist()
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"{label} {conf:.2f}", (x1, max(20, y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return annotated, counts

def annotate_haar(img: np.ndarray):
    cascade = load_haar()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    annotated = img.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(annotated, "Face", (x, max(20, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    return annotated, {"Face": int(len(faces))}

def encode_image(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()

@app.route("/")
def index():
    yolo_weights = [p.name for p in (MODELS_DIR.glob("*.pt"))]
    haar_files = [p.name for p in (MODELS_DIR.glob("*.xml"))]
    return render_template("index.html",
                           yolo_weights=yolo_weights,
                           haar_files=haar_files,
                           default_yolo=DEFAULT_YOLO_WEIGHTS,
                           default_haar=DEFAULT_HAAR_XML)

@app.post("/detect-image")
def detect_image():
    try:
        img_file = request.files.get("image")
        if img_file is None:
            return jsonify({"ok": False, "error": "No image uploaded"}), 400
        model_choice = request.form.get("model", "yolo")
        img = read_image(img_file)
        if model_choice == "yolo":
            annotated, counts = annotate_yolo(img)
        else:
            annotated, counts = annotate_haar(img)
        jpg = encode_image(annotated)
        return send_file(io.BytesIO(jpg), mimetype="image/jpeg")
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500

@app.post("/detect-frame")
def detect_frame():
    try:
        img_file = request.files.get("frame")
        if img_file is None:
            return jsonify({"ok": False, "error": "No frame"}), 400
        model_choice = request.form.get("model", "yolo")
        img = read_image(img_file)
        if model_choice == "yolo":
            annotated, counts = annotate_yolo(img)
        else:
            annotated, counts = annotate_haar(img)
        jpg = encode_image(annotated)
        import base64
        b64 = base64.b64encode(jpg).decode("utf-8")
        return jsonify({"ok": True, "image_base64": b64, "counts": counts})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/health")
def health():
    try:
        import importlib
        info = {
            "cwd": str(APP_ROOT),
            "models_dir_exists": MODELS_DIR.exists(),
            "models": {
                "pt": [p.name for p in MODELS_DIR.glob("*.pt")],
                "xml": [p.name for p in MODELS_DIR.glob("*.xml")],
            },
            "ultralytics_installed": YOLO is not None,
        }
        try:
            import torch
            info["torch"] = {
                "available": True,
                "cuda_available": bool(getattr(torch, "cuda", None) and torch.cuda.is_available()),
                "version": getattr(torch, "__version__", "unknown")
            }
        except Exception:
            info["torch"] = {"available": False}
        return jsonify({"ok": True, "info": info})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=7860, debug=True)
