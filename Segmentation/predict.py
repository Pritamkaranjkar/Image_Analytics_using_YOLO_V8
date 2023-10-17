from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-seg.pt')  # load an official model
model = YOLO('C:\\Users\\Pritam\\runs\\segment\\train2\\weights\\best.pt')  # load a custom model

# Predict with the model
results = model('E:\\@RUIA\\Data Science\\Internal Project\\Segmentation\\x-ray\\images\\train')  # predict on an image

