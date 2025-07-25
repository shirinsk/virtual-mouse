import cv2
import numpy as np
import dlib
from math import hypot
import pyglet
import time

# Load sounds
sound = pyglet.media.load("sound.wav", streaming=False)
left_sound = pyglet.media.load("left.wav", streaming=False)
right_sound = pyglet.media.load("right.wav", streaming=False)

cap = cv2.VideoCapture(0)
board = np.zeros((300, 1400), np.uint8)
board[:] = 255

# Face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Keyboard settings
keyboard = np.zeros((600, 1000, 3), np.uint8)

keys_set_1 = {
    0: "Q", 1: "W", 2: "E", 3: "R", 4: "T",
    5: "A", 6: "S", 7: "D", 8: "F", 9: "G",
    10: "2", 11: "X", 12: "C", 13: "V", 14: "<"
}

keys_set_2 = {
    0: "Y", 1: "U", 2: "I", 3: "O", 4: "P",
    5: "H", 6: "J", 7: "K", 8: "L", 9: "",
    10: "V", 11: "B", 12: "N", 13: "M", 14: "<"
}

def draw_letters(letter_index, text, letter_light):
    x = (letter_index % 5) * 200
    y = (letter_index // 5) * 200
    width = 200
    height = 200
    th = 3
    font_letter = cv2.FONT_HERSHEY_PLAIN
    font_scale = 10
    font_th = 4
    text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
    text_x = int((width - text_size[0]) / 2) + x
    text_y = int((height + text_size[1]) / 2) + y

    if letter_light:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (255, 255, 255), -1)
        cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (0, 0, 0), font_th)
    else:
        cv2.rectangle(keyboard, (x + th, y + th), (x + width - th, y + height - th), (51, 51, 51), -1)
        cv2.putText(keyboard, text, (text_x, text_y), font_letter, font_scale, (255, 255, 255), font_th)

def draw_menu():
    rows, cols, _ = keyboard.shape
    th_lines = 4
    cv2.line(keyboard, (cols // 2, 0), (cols // 2, rows), (51, 51, 51), th_lines)
    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(keyboard, "LEFT", (80, 300), font, 6, (255, 255, 255), 5)
    cv2.putText(keyboard, "RIGHT", (cols // 2 + 80, 300), font, 6, (255, 255, 255), 5)

def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_length / ver_line_length
    return ratio

def eyes_contour_points(facial_landmarks):
    left_eye = []
    right_eye = []
    for n in range(36, 42):
        x = facial_landmarks.part(n).x
        y = facial_landmarks.part(n).y
        left_eye.append([x, y])
    for n in range(42, 48):
        x = facial_landmarks.part(n).x
        y = facial_landmarks.part(n).y
        right_eye.append([x, y])
    return np.array(left_eye, np.int32), np.array(right_eye, np.int32)

def get_gaze_ratio(eye_points, facial_landmarks):
    eye_region = np.array([
        (facial_landmarks.part(point).x, facial_landmarks.part(point).y) for point in eye_points
    ], np.int32)

    mask = np.zeros((frame.shape[0], frame.shape[1]), np.uint8)
    cv2.polylines(mask, [eye_region], True, 255, 2)
    cv2.fillPoly(mask, [eye_region], 255)

    eye = cv2.bitwise_and(gray, gray, mask=mask)
    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])

    gray_eye = eye[min_y:max_y, min_x:max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)

    height, width = threshold_eye.shape
    left_side_white = cv2.countNonZero(threshold_eye[:, :width // 2])
    right_side_white = cv2.countNonZero(threshold_eye[:, width // 2:])

    if right_side_white == 0:
        return 5
    elif left_side_white == 0:
        return 1
    else:
        return left_side_white / right_side_white

# Counters and settings
frames = 0
letter_index = 0
blinking_frames = 0
frames_to_blink = 6
frames_active_letter = 9
text = ""
keyboard_selected = "left"
last_keyboard_selected = "left"
select_keyboard_menu = True
keyboard_selection_frames = 0
font = cv2.FONT_HERSHEY_PLAIN

while True:
    _, frame = cap.read()
    rows, cols, _ = frame.shape
    keyboard[:] = (26, 26, 26)
    frames += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame[rows - 50: rows, 0: cols] = (255, 255, 255)

    if select_keyboard_menu:
        draw_menu()

    if keyboard_selected == "left":
        keys_set = keys_set_1
    else:
        keys_set = keys_set_2

    active_letter = keys_set[letter_index]

    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        left_eye, right_eye = eyes_contour_points(landmarks)
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if select_keyboard_menu:
            gaze_ratio_left = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
            gaze_ratio_right = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
            gaze_ratio = (gaze_ratio_right + gaze_ratio_left) / 2

            if gaze_ratio <= 1:
                keyboard_selected = "right"
                keyboard_selection_frames += 1
                if keyboard_selection_frames == 15:
                    select_keyboard_menu = False
                    right_sound.play()
                    frames = 0
                    keyboard_selection_frames = 0
            else:
                keyboard_selected = "left"
                keyboard_selection_frames += 1
                if keyboard_selection_frames == 15:
                    select_keyboard_menu = False
                    left_sound.play()
                    frames = 0
                    keyboard_selection_frames = 0

        if blinking_ratio > 5:
            blinking_frames += 1
            if blinking_frames == frames_to_blink:
                if active_letter == "<":
                    text = text[:-1]
                else:
                    text += active_letter
                sound.play()
                select_keyboard_menu = True
                blinking_frames = 0
        else:
            blinking_frames = 0

    if not select_keyboard_menu:
        if frames == frames_active_letter:
            letter_index += 1
            frames = 0
        if letter_index >= len(keys_set):
            letter_index = 0
        for i in range(len(keys_set)):
            light = i == letter_index
            draw_letters(i, keys_set[i], light)

    cv2.putText(board, text, (80, 100), font, 9, 0, 3)
    percentage_blinking = blinking_frames / frames_to_blink
    loading_x = int(cols * percentage_blinking)
    cv2.rectangle(frame, (0, rows - 50), (loading_x, rows), (51, 51, 51), -1)

    cv2.imshow("Frame", frame)
    cv2.imshow("Virtual keyboard", keyboard)
    cv2.imshow("Board", board)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
