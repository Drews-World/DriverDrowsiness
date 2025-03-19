import cv2
import dlib
import numpy as np
import joblib
import time
import pygame
from imutils import face_utils

pygame.mixer.init()
alert_sound = pygame.mixer.Sound("alert.mp3")  


# Load trained model
model = joblib.load("drowsiness_model.pkl")

# Load dlib face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# Function to calculate mouth aspect ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[1] - mouth[7])  # Vertical distance (top-bottom)
    B = np.linalg.norm(mouth[2] - mouth[6])  # Vertical distance (mid)
    C = np.linalg.norm(mouth[0] - mouth[4])  # Horizontal distance (left-right)
    return (A + B) / (2.0 * C)

# Function to calculate head tilt ratio
def head_tilt_ratio(shape):
    nose = shape[33]  # Nose tip
    chin = shape[8]   # Chin
    left_cheek = shape[1]
    right_cheek = shape[15]

    vertical_dist = np.linalg.norm(nose - chin)
    horizontal_dist = np.linalg.norm(left_cheek - right_cheek)
    return vertical_dist / horizontal_dist 

# Start video capture
cap = cv2.VideoCapture(0)  

drowsy_start_time = None 
alert_playing = False  

drowsiness_scores = []
history_length = 15  

while True:
    
    # Initialize running averages
    ear_avg = None
    mar_avg = None
    htr_avg = None
    alpha = 0.1  
    
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        left_eye = shape[36:42]
        right_eye = shape[42:48]
        mouth = shape[60:68]

        # Compute EAR, MAR, and HTR
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        mar = mouth_aspect_ratio(mouth)
        htr = head_tilt_ratio(shape) 

        if ear_avg is None:
            ear_avg, mar_avg, htr_avg = ear, mar, htr 
        else:
            ear_avg = (1 - alpha) * ear_avg + alpha * ear
            mar_avg = (1 - alpha) * mar_avg + alpha * mar
            htr_avg = (1 - alpha) * htr_avg + alpha * htr

        # Adjust sensitivity of EAR & MAR
        adjusted_ear = ear_avg * 1.2   
        adjusted_mar = mar_avg * 0.5  

        drowsiness_score = model.predict([[adjusted_ear, adjusted_mar, htr]])[0]
      
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"HTR: {htr:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f"Drowsiness Score: {drowsiness_score:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        
        # Update the history buffer
        drowsiness_scores.append(drowsiness_score)
        if len(drowsiness_scores) > history_length:
            drowsiness_scores.pop(0) 

        # Compute moving average for smoother detection
        avg_drowsiness_score = np.mean(drowsiness_scores)

        cv2.putText(frame, f"Drowsiness Score: {avg_drowsiness_score:.2f}", (10, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Drowsiness alert triggers only if sustained over multiple frames
        if avg_drowsiness_score > 0.85:
            if drowsy_start_time is None:
                drowsy_start_time = time.time()
                alert_playing = False  
            elif time.time() - drowsy_start_time > 1.5:  
                if not alert_playing:  
                    pygame.mixer.Sound.play(alert_sound)
                    alert_playing = True  
                cv2.putText(frame, "DROWSY! WAKE UP!", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            drowsy_start_time = None  # Reset timer if not drowsy


     
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow("Driver Drowsiness Detection", frame)

  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()