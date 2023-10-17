from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')  

model.train(data='E:\\@RUIA\\Data Science\\Internal Project\\Classsification\\x-ray',epochs=1, imgsz=64)