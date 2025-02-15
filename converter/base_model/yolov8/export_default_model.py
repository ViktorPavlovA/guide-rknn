from ultralytics import YOLO

# Load the YOLO8 model
model = YOLO("yolov8n.pt")

# Export the model to ONNX format
model.export(format="onnx")  
