from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
results = model.train(data="E:\\@RUIA\\Data Science\\Internal Project\\Object_detection\\config.yaml", epochs=200, imgsz=640)  # train the model