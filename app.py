
from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained YOLOv8 model
MODEL_PATH = "vision2.pt"  # Change if needed
model = YOLO(MODEL_PATH)

@app.route('/detect', methods=['POST'])
def detect():
    """
    POST /detect endpoint.
    Accepts an image file and returns detections.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        # Read image from request
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({"error": "Invalid image format"}), 400

        # Run YOLOv8 detection
        results = model(image, conf=0.4)

        # Process detections
        detections = []
        for box in results[0].boxes:
            label = results[0].names[int(box.cls[0])]
            confidence = float(box.conf[0])
            bbox_xywh = box.xywh[0].tolist()  # x_center, y_center, width, height

            detections.append({
                "label": label,
                "confidence": round(confidence, 2),
                "bbox": [round(coord, 2) for coord in bbox_xywh]
            })

        # Final JSON response
        response = {
            "model": MODEL_PATH,
            "num_detections": len(detections),
            "detections": detections
        }

        return jsonify(response), 200

    except Exception as e:
        # Catch any unexpected errors
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print(f"🚀 Starting detection API with model: {MODEL_PATH}")
    app.run(host='0.0.0.0', port=5000)
