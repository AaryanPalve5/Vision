# app.py — FINAL version (ONNX + NMS=True)

from flask import Flask, request, jsonify
import onnxruntime as ort
import numpy as np
from PIL import Image

# Load ONNX model
session = ort.InferenceSession("vision2.onnx")  # Use your exported ONNX with NMS=True
input_name = session.get_inputs()[0].name

# Class names — from your data.yaml
class_names = {
    0: "door",
    1: "window"
}

# Initialize Flask app
app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect():
    # Load uploaded image
    file = request.files['file']
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
