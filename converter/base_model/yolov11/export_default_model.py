from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.export(format="onnx") 
