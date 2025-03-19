import numpy as np

# Load labels
y = np.load("y_labels.npy")

# Count occurrences of each class
drowsy_count = np.sum(y == 1)
awake_count = np.sum(y == 0)

print(f"Drowsy Samples: {drowsy_count}")
print(f"Awake Samples: {awake_count}")