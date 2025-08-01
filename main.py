import cv2
import mediapipe as mp
import time
import numpy as np
from controlkeys import right_pressed, left_pressed, up_pressed, down_pressed
from controlkeys import KeyOn, KeyOff

# Wait a moment for the camera to warm up
time.sleep(2.0)

# Track currently pressed keys
current_key_pressed = set()

# MediaPipe setup
mp_draw = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands
tipIds = [4, 8, 12, 16, 20]

# Start webcam
video = cv2.VideoCapture(0)

# Main hand tracking loop
with mp_hand.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while True:
        keyPressed = False
        ret, image = video.read()
        if not ret:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        lmList = []
        text = ''

        # Detect hand and landmarks
        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                label = results.multi_handedness[idx].classification[0].label
                text = label

                h, w, c = image.shape
                for id, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])

                mp_draw.draw_landmarks(image, hand_landmarks, mp_hand.HAND_CONNECTIONS)

        # Gesture recognition
        fingers = []
        if lmList:
            # Thumb
            fingers.append(1 if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1] else 0)
            # Fingers
            for id in range(1, 5):
                fingers.append(1 if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2] else 0)

            total = fingers.count(1)
            key_pressed = None

            if total == 4 and text == "Right":
                cv2.rectangle(image, (400, 300), (600, 425), (255, 255, 255), cv2.FILLED)
                cv2.putText(image, "LEFT", (400, 375), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
                KeyOn(left_pressed)
                key_pressed = left_pressed
                keyPressed = True

            elif total == 5 and text == "Left":
                cv2.rectangle(image, (400, 300), (600, 425), (255, 255, 255), cv2.FILLED)
                cv2.putText(image, "RIGHT", (400, 375), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
                KeyOn(right_pressed)
                key_pressed = right_pressed
                keyPressed = True

            elif total == 1:
                cv2.rectangle(image, (400, 300), (600, 425), (255, 255, 255), cv2.FILLED)
                cv2.putText(image, "UP", (400, 375), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
                KeyOn(up_pressed)
                key_pressed = up_pressed
                keyPressed = True

            elif total == 0:
                cv2.rectangle(image, (400, 300), (600, 425), (255, 255, 255), cv2.FILLED)
                cv2.putText(image, "DOWN", (400, 375), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
                KeyOn(down_pressed)
                key_pressed = down_pressed
                keyPressed = True

            if keyPressed:
                if key_pressed not in current_key_pressed:
                    for key in current_key_pressed.copy():
                        KeyOff(key)
                    current_key_pressed.clear()
                current_key_pressed.add(key_pressed)

        # No key pressed → release all
        if not keyPressed and current_key_pressed:
            for key in current_key_pressed.copy():
                KeyOff(key)
            current_key_pressed.clear()

        # Display the video frame
        cv2.imshow("Hand Gesture Control", image)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
video.release()
cv2.destroyAllWindows()

