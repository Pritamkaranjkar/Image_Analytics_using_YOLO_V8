from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n.pt')  # load an official model
model = YOLO('C:\\Users\\Pritam\\runs\\detect\\train2\\weights\\last.pt')  # load a custom model


# Predict with the model
results = model('E:\\@RUIA\\Data Science\\Internal Project\\Object_detection\\x-ray\\images\\train\\IMG0000057.jpg')  # predict on an image
print("Prediction:")
# print(results)