# detect_image.py

from ultralytics import YOLO
import cv2
import json

# Load the trained model
model = YOLO("vision2.pt")

# Path to the image you want to detect
image_path = "Raw dataset/16.png"  # CHANGE THIS to your image

# Run detection
results = model(image_path, save=True, conf=0.4)  # save=True will save results in runs/detect

# Extract detections and format them
detections = []
for box in results[0].boxes:
    label = results[0].names[int(box.cls[0])]
    confidence = float(box.conf[0])
    # box.xywh[0] gives (x_center, y_center, width, height), convert to list
    bbox_xywh = box.xywh[0].tolist()
    detections.append({
        "label": label,
        "confidence": round(confidence, 2),
        "bbox": bbox_xywh
    })

# Prepare final output
output = {
    "detections": detections
}

# Print as JSON
print(json.dumps(output, indent=2))

# Optional: Display the image with detections
result_image = results[0].plot()
cv2.imshow("Detection", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Detection complete.")
