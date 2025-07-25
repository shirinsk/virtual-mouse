import cv2
import mediapipe as mp
import pyautogui

# Initialize camera and face mesh detector
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

sensitivity_factor = 1.5  

while True:
    success, frame = cam.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    
    if output.multi_face_landmarks:
        for face_landmarks in output.multi_face_landmarks:
            frame_h, frame_w, _ = frame.shape
            landmarks = face_landmarks.landmark
            
            # Eye tracking (landmarks 474 & 475)
            for id, landmark in enumerate([landmarks[474], landmarks[475]]):
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                
                if id == 1:  # Move mouse pointer based on second point
                    screen_x = int(landmark.x * screen_w * sensitivity_factor)
                    screen_y = int(landmark.y * screen_h * sensitivity_factor)
                    pyautogui.moveTo(screen_x, screen_y, duration=0.05)  # Smooth movement
            
            # Blink Detection for Click (Landmarks 145 & 159)
            left_eye_top = landmarks[145].y
            left_eye_bottom = landmarks[159].y
            cv2.circle(frame, (int(landmarks[145].x * frame_w), int(left_eye_top * frame_h)), 3, (0, 255, 255), -1)
            cv2.circle(frame, (int(landmarks[159].x * frame_w), int(left_eye_bottom * frame_h)), 3, (0, 255, 255), -1)

            if abs(left_eye_top - left_eye_bottom) < 0.010:
                print('Click')
                pyautogui.click()
                pyautogui.sleep(0.5)  # Reduced sleep time for faster response

    cv2.imshow('Eye Controller', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
