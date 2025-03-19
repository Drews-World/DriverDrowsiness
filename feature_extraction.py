import cv2
import dlib
import numpy as np
from imutils import face_utils

# Load dlib’s face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/drewhumes/Desktop/Projects/AI/DriverDrowsiness/shape_predictor_68_face_landmarks.dat")

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# Function to calculate mouth aspect ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[13] - mouth[19])  # Vertical distance upper-lower lip
    B = np.linalg.norm(mouth[15] - mouth[17])  # Vertical distance upper-lower lip (inner)
    C = np.linalg.norm(mouth[0] - mouth[6])  # Horizontal distance (mouth width)
    return (A + B) / (2.0 * C)

def head_tilt_ratio(shape):
    nose = shape[33] 
    chin = shape[8]   
    left_cheek = shape[1]
    right_cheek = shape[15]

    vertical_dist = np.linalg.norm(nose - chin)
    horizontal_dist = np.linalg.norm(left_cheek - right_cheek)
    return vertical_dist / horizontal_dist 

# Load dataset
X = np.load("X_data.npy")  # Preprocessed images
y = np.load("y_labels.npy")  # Labels

features = []

# Process each image in dataset
for img in X:
    gray = (img * 255).astype(np.uint8)  
    rects = detector(gray, 0)

    if len(rects) > 0:
        shape = predictor(gray, rects[0])
        shape = face_utils.shape_to_np(shape)

        left_eye = shape[36:42]
        right_eye = shape[42:48]
        mouth = shape[48:68]

        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        mar = mouth_aspect_ratio(mouth)
        head_tilt = head_tilt_ratio(shape)
        
        features.append([ear, mar, head_tilt])


# Convert features to NumPy array and save
features = np.array(features)
np.save("features.npy", features)

print("Feature extraction complete! Data saved as features.npy")
