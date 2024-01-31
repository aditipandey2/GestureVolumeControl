import cv2
import mediapipe as mp
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize Volume Control Library
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
min_vol, max_vol, vol_bar, vol_percentage = vol_range[0], vol_range[1], 400, 0

# Webcam Setup
w_cam, h_cam = 640, 480
cam = cv2.VideoCapture(0)
cam.set(3, w_cam)
cam.set(4, h_cam)

# Initialize Mediapipe Hand Landmark Model
with mp.solutions.hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cam.isOpened():
        success, image = cam.read()

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style()
                )

        # Find position of Hand landmarks
        lm_list = []
        if results.multi_hand_landmarks:
            my_hand = results.multi_hand_landmarks[0]
            for id, lm in enumerate(my_hand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])

        # Assign variables for Thumb and Index finger position
        if len(lm_list) != 0:
            x1, y1 = lm_list[4][1], lm_list[4][2]
            x2, y2 = lm_list[8][1], lm_list[8][2]

            # Mark Thumb and Index finger
            cv2.circle(image, (x1, y1), 15, (255, 255, 255))
            cv2.circle(image, (x2, y2), 15, (255, 255, 255))
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

            length = math.hypot(x2 - x1, y2 - y1)
            if length < 50:
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 3)

            vol = np.interp(length, [50, 220], [min_vol, max_vol])
            volume.SetMasterVolumeLevel(vol, None)
            vol_bar = np.interp(length, [50, 220], [400, 150])
            vol_percentage = np.interp(length, [50, 220], [0, 100])

            # Volume Bar
            cv2.rectangle(image, (50, 150), (85, 400), (0, 0, 0), 3)
            cv2.rectangle(image, (50, int(vol_bar)), (85, 400), (0, 0, 0), cv2.FILLED)
            cv2.putText(image, f'{int(vol_percentage)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                        1, (0, 0, 0), 3)

        cv2.imshow('Hand Detector', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()
