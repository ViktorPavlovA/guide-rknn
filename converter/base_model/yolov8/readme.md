#check information about two model


1. Source about yolo11rknn 

`wget https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/yolov8/yolov8n.onnx`

2. Download from ultralytics

```
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.export(format="onnx")  

```


