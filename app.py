# app.py — MERGED version: Web UI + /detect API + ONNX model

from flask import Flask, render_template, request, url_for, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw
import os
import numpy as np
import onnxruntime as ort

# Initialize Flask app
app = Flask(__name__)

# Ensure folders exist
os.makedirs("uploads", exist_ok=True)
os.makedirs("static/results", exist_ok=True)

# Load ONNX model
session = ort.InferenceSession("vision2.onnx")  # Your ONNX with NMS=True
input_name = session.get_inputs()[0].name

# Class names — from your data.yaml
class_names = {
    0: "door",
    1: "window"
}

# ONNX inference function
def run_detection(image_path):
    # Load image
    image = Image.open(image_path).convert("RGB")
    image_resized = image.resize((640, 640))  # Match ONNX input size
    image_np = np.array(image_resized).transpose(2, 0, 1)  # HWC -> CHW
    image_np = np.expand_dims(image_np, axis=0).astype(np.float32) / 255.0

    # Run ONNX
    outputs = session.run(None, {input_name: image_np})
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
    return detections

# Draw bounding boxes
def draw_boxes(image_path, detections, output_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Define colors for each label
    label_colors = {
        "window": "blue",
        "door": "green"
    }

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        label = det['label']
        confidence = det['confidence']

        # Pick color for this label (default = red if unknown)
        color = label_colors.get(label, "red")

        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

        # Draw label text
        text = f"{label} ({confidence:.2f})"
        text_position = (x1 + 5, y1 - 20 if y1 - 20 > 0 else y1 + 5)

        draw.text(text_position, text, fill=color)

    image.save(output_path)

# Web UI route — Upload Form
@app.route("/", methods=["GET", "POST"])
def index():
    detections = []
    error = None
    result_image_path = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = secure_filename(file.filename)
            input_path = os.path.join("uploads", filename)
            file.save(input_path)

            # Run ONNX detection
            detections = run_detection(input_path)

            # Draw and save result image
            output_filename = f"boxed_{filename}"
            output_path = os.path.join("static/results", output_filename)
            draw_boxes(input_path, detections, output_path)

            result_image_path = url_for('static', filename=f"results/{output_filename}")

    return render_template("index.html", detections=detections, error=error, result_image=result_image_path)

# API route — /detect (POST)
@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'GET':
        return "ONNX /detect endpoint is ready (send POST with image)."

    if 'file' not in request.files:
        return jsonify(error="No file part in the request."), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify(error="No selected file."), 400

    try:
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        input_path = os.path.join("uploads", filename)
        file.save(input_path)

        # Run ONNX detection
        detections = run_detection(input_path)

        # Return JSON detections
        return jsonify(detections=detections)

    except Exception as e:
        return jsonify(error=str(e)), 500

# Run app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
