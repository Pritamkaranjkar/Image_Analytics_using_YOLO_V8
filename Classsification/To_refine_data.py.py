import cv2 #pip install opencv-python
import os
import numpy as np #pip install numpy
import matplotlib.pyplot as plt #pip install matplotlib

# Prompt the user to enter the path to the dataset
dataset_path = input("Enter the path to the dataset: ")

# Load the images into a list
images = []
for image_path in os.listdir(dataset_path):
    image = cv2.imread(os.path.join(dataset_path, image_path))
    images.append(image)

# # Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply the Canny edge detector
edges = cv2.Canny(gray, 50, 150)

# Display the edge image
cv2.imshow("Edges", edges)
cv2.waitKey(0)

# Apply the median filter
denoised_image = cv2.medianBlur(image, 5)

# Display the denoised image
cv2.imshow("Denoised Image", denoised_image)
cv2.waitKey(0)

# Apply the Otsu thresholding algorithm
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Find contours in the image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours on the image
for contour in contours:
    cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)

# Display the segmented image
cv2.imshow("Segmented Image", image)
cv2.waitKey(0)

#histogram
# Calculate the color histogram for each image
image_histograms = []
for image in images:
    histogram = cv2.calcHist([image], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    image_histograms.append(histogram)

mean_image_histogram = np.mean(image_histograms, axis=0).flatten()

# Display the mean image histogram
plt.plot(mean_image_histogram)
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.title("Mean Image Histogram")
plt.show()
