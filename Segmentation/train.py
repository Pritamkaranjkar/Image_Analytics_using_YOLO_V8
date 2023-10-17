from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)

model.train(data='E:\\@RUIA\\Data Science\\Internal Project\\Segmentation\\config.yaml', epochs=1, imgsz=640)