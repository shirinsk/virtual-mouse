import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# Initialize PyAutoGUI
pyautogui.FAILSAFE = False  

# Initialize MediaPipe Face Mesh & Hands
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Get screen size
screen_width, screen_height = pyautogui.size()

# Cursor settings
calibrated_center = None  
sensitivity_factor = 7
smoothing_factor = 0.5

# Click detection settings
blink_threshold = 0.015  
blink_time = time.time()  

# Start webcam
cap = cv2.VideoCapture(0)
last_x, last_y = screen_width // 2, screen_height // 2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results_face = face_mesh.process(rgb_frame)
    results_hands = hands.process(rgb_frame)

    frame_height, frame_width, _ = frame.shape

    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            # Get eye landmarks
            left_eye_indices = [33, 133, 159, 145]
            right_eye_indices = [362, 263, 386, 374]

            left_eye_points = np.array([(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in left_eye_indices])
            right_eye_points = np.array([(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in right_eye_indices])

            # Eye centers
            left_eye_center = np.mean(left_eye_points, axis=0)
            right_eye_center = np.mean(right_eye_points, axis=0)

            # Draw small green dots on eye centers
            lx, ly = int(left_eye_center[0] * frame_width), int(left_eye_center[1] * frame_height)
            rx, ry = int(right_eye_center[0] * frame_width), int(right_eye_center[1] * frame_height)
            cv2.circle(frame, (lx, ly), 3, (0, 255, 0), -1)
            cv2.circle(frame, (rx, ry), 3, (0, 255, 0), -1)

            # Average the two eye centers
            gaze_x = (left_eye_center[0] + right_eye_center[0]) / 2
            gaze_y = (left_eye_center[1] + right_eye_center[1]) / 2

            # Calibrate on first frame
            if calibrated_center is None:
                calibrated_center = (gaze_x, gaze_y)

            # Cursor movement
            delta_x = (gaze_x - calibrated_center[0]) * screen_width * sensitivity_factor
            delta_y = (gaze_y - calibrated_center[1]) * screen_height * sensitivity_factor

            screen_x = int(last_x + smoothing_factor * (delta_x - last_x))
            screen_y = int(last_y + smoothing_factor * (delta_y - last_y))

            screen_x = max(5, min(screen_x, screen_width - 5))
            screen_y = max(5, min(screen_y, screen_height - 5))

            pyautogui.moveTo(screen_x, screen_y, duration=0.05)
            last_x, last_y = screen_x, screen_y

            # Blink detection
            left_eye_height = abs(left_eye_points[2][1] - left_eye_points[3][1])
            if left_eye_height < blink_threshold and (time.time() - blink_time) > 0.5:
                pyautogui.click()
                blink_time = time.time()

    cv2.imshow("Eye Tracking & Hand Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
