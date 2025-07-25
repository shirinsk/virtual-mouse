import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import joblib  # For loading the trained model

# Load the trained model
model = joblib.load("eye_tracking_model.pkl")

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Get screen size
screen_width, screen_height = pyautogui.size()

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip for natural movement
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results_face = face_mesh.process(rgb_frame)

    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            # Get iris landmarks
            left_iris = [face_landmarks.landmark[i] for i in [468, 469, 470, 471, 472]]
            right_iris = [face_landmarks.landmark[i] for i in [473, 474, 475, 476, 477]]

            # Convert landmarks to numpy arrays
            left_iris_center = np.mean([(p.x, p.y) for p in left_iris], axis=0)
            right_iris_center = np.mean([(p.x, p.y) for p in right_iris], axis=0)

            # Average iris position
            iris_x = (left_iris_center[0] + right_iris_center[0]) / 2
            iris_y = (left_iris_center[1] + right_iris_center[1]) / 2

            # Predict cursor position using trained model
            predicted_cursor = model.predict([[iris_x, iris_y]])
            screen_x, screen_y = int(predicted_cursor[0][0]), int(predicted_cursor[0][1])

            # Keep cursor within screen bounds
            screen_x = max(5, min(screen_x, screen_width - 5))
            screen_y = max(5, min(screen_y, screen_height - 5))

            # Move cursor
            pyautogui.moveTo(screen_x, screen_y, duration=0.05)

    # Display output
    cv2.imshow("Iris Tracking Mouse Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
