# Hand Gesture Virtual Mouse & Volume Control

Control your mouse cursor and system volume using hand gestures via your webcam — no physical input needed.

---

## Requirements

- Python 3.8+
- A webcam
- Windows OS (uses `pycaw` for audio control)

---

## Installation

Install all dependencies with:

```bash
pip install opencv-python mediapipe pyautogui pycaw comtypes numpy
```

| Library | Purpose |
|---|---|
| `opencv-python` | Webcam capture and image rendering |
| `mediapipe` | Hand landmark detection |
| `pyautogui` | Mouse movement and click control |
| `pycaw` | Windows system volume control |
| `comtypes` | Required by pycaw |
| `numpy` | Numerical interpolation |

You also need the `hand_landmarker.task` model file in the project root (already included).

---

## How to Run

```bash
python cursor.py
```

Press `q` or make a **fist and hold** to exit.

---

## Gesture Guide

### Cursor Mode (default)

| Gesture | Action |
|---|---|
| Index finger up | Move mouse cursor |
| All 5 fingers open (full hand) | Left click |
| Index + Middle fingers up — hold 0.5s | Switch to Volume Mode |
| Fist (all fingers closed) — hold 0.8s | Exit the program |

### Volume Mode

| Gesture | Action |
|---|---|
| Pinch thumb & index finger | Control system volume (closer = lower, wider = louder) |
| Index + Middle fingers up — hold 0.5s | Switch back to Cursor Mode |
| Fist (all fingers closed) — hold 0.8s | Exit the program |

> The current mode is displayed on screen. You can also press `m` on your keyboard to toggle modes.

---

## Notes

- Make sure your hand is clearly visible and well-lit for best tracking accuracy.
- The app uses your default webcam (index 0).
- `pyautogui.FAILSAFE` is disabled — move your mouse to a screen corner won't abort the script.
