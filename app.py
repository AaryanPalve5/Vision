# app.py — FINAL version (ONNX + NMS=True) + GET/POST /detect

from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
from PIL import Image

# Load ONNX model
session = ort.InferenceSession("vision2.onnx")  # Correct ONNX with NMS=True
input_name = session.get_inputs()[0].name

# Class names — from your data.yaml
class_names = {
    0: "door",
    1: "window"
}

# Initialize Flask app
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return "ONNX Runtime Vision API is running."

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'GET':
        return "ONNX /detect endpoint is ready (send POST with image)."

    # POST method — process image
    if 'file' not in request.files:
        return jsonify(error="No file part in the request."), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No selected file."), 400

    try:
        # Load uploaded image
        image = Image.open(file).convert("RGB")
        image = image.resize((640, 640))  # Must match ONNX export imgsz!
        image_np = np.array(image).transpose(2, 0, 1)  # HWC -> CHW
        image_np = np.expand_dims(image_np, axis=0).astype(np.float32) / 255.0

        # Run ONNX inference
        outputs = session.run(None, {input_name: image_np})

        # Remove batch dimension
        output_array = outputs[0][0]  # [num_detections, 6]

        # Parse detections
        detections = []
        for det in output_array:
            x1, y1, x2, y2, conf, class_id = det.tolist()
            confidence = float(conf)
            if confidence > 0.4:  # Confidence threshold
                label = class_names.get(int(class_id), f"class_{int(class_id)}")
                bbox = [x1, y1, x2, y2]
                detections.append({
                    "label": label,
                    "confidence": round(confidence, 2),
                    "bbox": bbox
                })

        # Return JSON response
        return jsonify(detections=detections)

    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
