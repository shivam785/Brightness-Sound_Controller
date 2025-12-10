````markdown
# Gesture Control: Brightness & Volume

Control system **brightness** and **audio volume** using hand gestures in real time with MediaPipe and OpenCV.

---

## Features
- Right hand controls **screen brightness**
- Left hand controls **system volume**
- Live on-screen bars for both levels
- Smooth adjustment using gesture distance
- Works with any webcam

---

## Requirements
- Windows OS (required for PyCAW audio control)
- Python 3.8+

### Python Libraries
```bash
pip install opencv-python mediapipe numpy comtypes pycaw screen_brightness_control
````

---

## How It Works

| Hand  | Gesture                 | Controls   |
| ----- | ----------------------- | ---------- |
| Right | Thumb ↔ Index distance  | Brightness |
| Left  | Thumb ↔ Middle distance | Volume     |

Brightness and volume scale based on the distance between fingertips.

---

## Run

```bash
python gesture_control.py
```

Press **Q** to exit.

---

## Controls Display

* Left side bar → Brightness percentage (0–100)
* Right side bar → Volume percentage (0–100)
* Hand landmarks shown live

---

## Notes

* Good lighting improves tracking accuracy
* Keep hands in frame for smoother control
* Sudden movement may cause minor jumps

---

## Files

* `gesture_control.py` — main controller script

---
