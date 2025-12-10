import cv2
import mediapipe as mp
import numpy as np
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbc

# ======================
#  Audio setup (pycaw)
# ======================
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volMin, volMax = volume.GetVolumeRange()[:2]  # dB range

# ======================
#  Mediapipe Hands
# ======================
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mpDraw = mp.solutions.drawing_utils

# ======================
#  Video Capture
# ======================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# ======================
#  GUI helper
# ======================
def draw_bar(img, x, y, w, h, value, max_value, label):
    """
    Draw a vertical bar with a label and percentage text.
    value: current value (0 - max_value)
    """
    value = max(0, min(max_value, value))  # clamp
    ratio = value / max_value if max_value > 0 else 0

    # Background bar
    cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 50), 2)

    # Filled bar (bottom -> up)
    filled_height = int(h * ratio)
    cv2.rectangle(
        img,
        (x, y + h - filled_height),
        (x + w, y + h),
        (0, 255, 0),
        -1
    )

    # Label
    cv2.putText(img, f"{label}", (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Percentage text
    cv2.putText(img, f"{int(value)}%",
                (x - 5, y + h + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

# ======================
#  State / smoothing
# ======================
alpha = 0.3  # smoothing factor
brightness_val = 50
volume_percent = 50

prev_brightness = brightness_val
prev_volume_db = np.interp(volume_percent, [0, 100], [volMin, volMax])

# Set initial system states
try:
    sbc.set_brightness(brightness_val)
except Exception:
    pass

try:
    volume.SetMasterVolumeLevel(prev_volume_db, None)
except Exception:
    pass

# ======================
#  Main loop
# ======================
while True:
    success, img = cap.read()
    if not success:
        print("[WARN] Failed to grab frame")
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                              results.multi_handedness):
            label = handedness.classification[0].label  # 'Right' or 'Left'

            # Collect landmarks
            lmList = []
            for lm in hand_landmarks.landmark:
                lmList.append([int(lm.x * w), int(lm.y * h)])

            # Draw hand landmarks
            mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)

            thumb_tip  = lmList[4]
            index_tip  = lmList[8]
            middle_tip = lmList[12]

            # ======================
            #  RIGHT HAND = BRIGHTNESS
            #  (Thumb–Index distance)
            # ======================
            if label == "Right":
                b_dist = hypot(index_tip[0] - thumb_tip[0],
                               index_tip[1] - thumb_tip[1])
                brightness_raw = np.interp(b_dist, [20, 200], [0, 100])

                # Smooth brightness
                prev_brightness = (1 - alpha) * prev_brightness + alpha * brightness_raw
                brightness_val = int(prev_brightness)

                try:
                    sbc.set_brightness(brightness_val)
                except Exception:
                    cv2.putText(img, "Brightness error",
                                (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2)

            # ======================
            #  LEFT HAND = VOLUME
            #  (Thumb–Middle distance)
            # ======================
            if label == "Left":
                v_dist = hypot(middle_tip[0] - thumb_tip[0],
                               middle_tip[1] - thumb_tip[1])
                vol_db_raw = np.interp(v_dist, [20, 200], [volMin, volMax])

                # Smooth volume
                prev_volume_db = (1 - alpha) * prev_volume_db + alpha * vol_db_raw

                try:
                    volume.SetMasterVolumeLevel(prev_volume_db, None)
                    volume_percent = int(
                        np.interp(prev_volume_db, [volMin, volMax], [0, 100])
                    )
                except Exception:
                    cv2.putText(img, "Volume error",
                                (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2)

    # ======================
    #  GUI OVERLAY
    # ======================

    # Semi-transparent bottom strip for title
    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

    cv2.putText(
        img,
        "Right hand: Brightness (Thumb-Index) | Left hand: Volume (Thumb-Middle)",
        (20, h - 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        (255, 255, 255), 2
    )

    # Brightness bar (left)
    draw_bar(
        img,
        x=60, y=100,
        w=40, h=300,
        value=brightness_val,
        max_value=100,
        label="Brightness"
    )

    # Volume bar (right)
    draw_bar(
        img,
        x=w - 100, y=100,
        w=40, h=300,
        value=volume_percent,
        max_value=100,
        label="Volume"
    )

    cv2.imshow("Gesture Control - Brightness (Right) & Volume (Left)", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
