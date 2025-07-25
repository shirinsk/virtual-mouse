import cv2
import mediapipe as mp
import pandas as pd
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Create a DataFrame to store data
data = []

# Start webcam
cap = cv2.VideoCapture(0)

print("Collecting data... Press 'q' to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror for natural movement
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_iris = [face_landmarks.landmark[i] for i in [468, 469, 470, 471, 472]]
            right_iris = [face_landmarks.landmark[i] for i in [473, 474, 475, 476, 477]]

            # Calculate iris center
            left_iris_center = [sum(p.x for p in left_iris) / 5, sum(p.y for p in left_iris) / 5]
            right_iris_center = [sum(p.x for p in right_iris) / 5, sum(p.y for p in right_iris) / 5]

            # Take the average of both iris positions
            eye_x = (left_iris_center[0] + right_iris_center[0]) / 2
            eye_y = (left_iris_center[1] + right_iris_center[1]) / 2

            # Store data
            data.append([eye_x, eye_y])
    
    cv2.imshow("Eye Tracking Data Collection", frame)
    
    # Stop
