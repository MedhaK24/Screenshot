from cv2 import cvtColor
import pyautogui
import cv2
import mediapipe as mp
import numpy as np
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
camera = cv2.VideoCapture(0)
fingertips = [8, 12, 16, 20]
thumbtip = 4


def takescreenshot(img, hand_landmarks):
    if hand_landmarks:
        for hand_landmark in hand_landmarks:
            lmlist = []
            for id, lm in enumerate(hand_landmark.landmark):
                lmlist.append(lm)
            fingerfullstatus = []
            for tip in fingertips:
                if lmlist[tip].x < lmlist[tip-3].x:
                    fingerfullstatus.append(True)
                else:
                    fingerfullstatus.append(False)
            if all(fingerfullstatus):
                image = pyautogui.screenshot()
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                cv2.imwrite('screenshot.png', image)
            mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS, mp_draw.DrawingSpec(
                (0, 0, 255), 2, 2), mp_draw.DrawingSpec((0, 255, 0), 4, 2))


while True:
    ret, img = camera.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    results = hands.process(img)
    hand_landmarks = results.multi_hand_landmarks
    takescreenshot(img, hand_landmarks)
    cv2.imshow('camera', img)
    cv2.waitKey(1)
