# detect_image.py

from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO("vision2.pt")

# Path to the image you want to detect
image_path = "Raw dataset/16.png"  # CHANGE THIS to your image

# Run detection
results = model(image_path, save=True, conf=0.4)  # save=True will save results in runs/detect

# Optional: Display the image with detections
result_image = results[0].plot()
cv2.imshow("Detection", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Detection complete.")
