# Hand Gesture Virtual Mouse & Volume Control

Control your mouse cursor, system volume, and type using a virtual keyboard — all through hand gestures via your webcam, no physical input needed.

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
| `pyautogui` | Mouse movement, click, and keyboard input |
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

## Modes

The app cycles through 3 modes. Switch between them using the **index + middle finger hold** gesture or press `m` on your keyboard.

```
Cursor Mode  →  Volume Mode  →  Keyboard Mode  →  (back to Cursor)
```

The current mode is always shown on screen.

---

## Gesture Guide

### Universal Gestures (work in all modes)

| Gesture | Action |
|---|---|
| Index + Middle fingers up — hold 0.5s | Cycle to next mode |
| Fist (all fingers closed) — hold 0.8s | Exit the program |
| `m` key on keyboard | Cycle to next mode |
| `q` key on keyboard | Exit the program |

---

### Cursor Mode (default)

| Gesture | Action |
|---|---|
| Index finger up | Move mouse cursor |
| All 5 fingers open (full hand) | Left click |

---

### Volume Mode

| Gesture | Action |
|---|---|
| Vary distance between thumb & index finger | Control system volume (pinch = lower, spread = louder) |

A volume bar is shown on screen with the current percentage.

---

### Keyboard Mode

A full virtual keyboard is rendered on the webcam feed. Point your index finger to hover over keys — the hovered key highlights green.

| Gesture | Action |
|---|---|
| Index finger hover | Highlight a key |
| All 5 fingers open (full hand) | Press the highlighted key |
| Hover + open hand on `SHIFT` | Toggle Shift on/off (one-shot — auto-disables after one letter) |
| Hover + open hand on `SPACE` | Type a space |
| Hover + open hand on `BACK` | Backspace |
| Hover + open hand on `ENTER` | Enter / newline |
| Open hand with no key highlighted | Mouse click (to focus a text box) |

#### Keyboard Layout

```
1  2  3  4  5  6  7  8  9  0
Q  W  E  R  T  Y  U  I  O  P
A  S  D  F  G  H  J  K  L
SHIFT  Z  X  C  V  B  N  M  BACK
      SPACE          ENTER
```

> In Keyboard Mode the OS cursor still moves with your index finger, so you can click into any text field before typing.

---

## Notes

- Make sure your hand is clearly visible and well-lit for best tracking accuracy.
- The app uses your default webcam (index 0).
- `pyautogui.FAILSAFE` is disabled — moving your mouse to a screen corner won't abort the script.
