import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Define paths
DATASET_PATH = "/Users/drewhumes/Desktop/Projects/AI/DriverDrowsiness/DDDS/train/"
OUTPUT_SIZE = (64, 64)  # Resize images for consistency

# Function to preprocess images
def preprocess_images(input_folder):
    images = []
    labels = []

    # Loop through both categories
    for category in ["drowsy", "awake"]:
        folder_path = os.path.join(input_folder, category)
        label = 1 if category == "drowsy" else 0  # 1 = Drowsy, 0 = Awake

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                continue  # Skip unreadable images

            # Convert to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Resize
            img = cv2.resize(img, OUTPUT_SIZE)

            # Normalize pixel values (0 to 1)
            img = img / 255.0  

            # Append to dataset
            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)

# Preprocess and save
X, y = preprocess_images(DATASET_PATH)
np.save("X_data.npy", X)  # Save images
np.save("y_labels.npy", y)  # Save labels

print("Preprocessing complete! Data saved as X_data.npy and y_labels.npy")