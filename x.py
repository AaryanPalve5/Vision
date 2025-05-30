from ultralytics import YOLO

model = YOLO("vision2.pt")
model.export(format="onnx", imgsz=640, nms=True, name="vision2_final.onnx")
