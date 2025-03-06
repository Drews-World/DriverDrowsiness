import matplotlib.pyplot as plt
import cv2
import os
import numpy as np

# Load preprocessed data
X = np.load("X_data.npy")
y = np.load("y_labels.npy")

# Display a sample image
plt.imshow(X[0], cmap="gray")
plt.title("Label: " + ("Drowsy" if y[0] == 1 else "Awake"))
plt.show()