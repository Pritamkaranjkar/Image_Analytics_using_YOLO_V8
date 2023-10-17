from ultralytics import YOLO

import numpy as np


model = YOLO('C:\\Users\\Pritam\\runs\\classify\\train\\weights\\best.pt')  

results = model('E:\\@RUIA\\Data Science\\Internal Project\\Classsification\\x-ray\\train\\Bacterial\\b1.jpeg')  # predict on an image

names_dict = results[0].names

probs = results[0].probs.data.tolist()

print(names_dict)
print(probs)
print('Prediction:')
print(names_dict[np.argmax(probs)])