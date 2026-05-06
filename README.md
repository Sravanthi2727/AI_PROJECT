# Hand Gesture Virtual Mouse & Controls

A single Windows application (`cursor.py`) that uses your **webcam** and **MediaPipe** hand tracking to control the **mouse**, **system volume**, **display brightness**, a **virtual keyboard**, and an **on-camera drawing canvas**—mostly without touching the keyboard or mouse.

---

## Table of contents

1. [What it does](#what-it-does-feature-summary)  
2. [Requirements](#requirements)  
3. [Hardware Components](#hardware-components)  
4. [Hardware Connections](#hardware-connections)  
5. [ESP32 LED Display Integration](#esp32-led-display-integration)  
6. [Installation](#installation)  
7. [How to run](#how-to-run)  
8. [Running with ESP32 Display](#running-with-esp32-display)  
9. [Finger notation](#finger-notation)  
10. [Modes and switching](#modes-and-switching)  
11. [Universal gestures](#universal-gestures-all-modes)  
12. [Cursor mode](#cursor-mode)  
13. [Volume mode](#volume-mode)  
14. [Brightness mode](#brightness-mode)  
15. [Keyboard mode](#keyboard-mode)  
16. [Drawing mode](#drawing-mode)  
17. [On-screen UI](#on-screen-ui)  
18. [Technical details](#technical-details-implementation)  
19. [Troubleshooting](#troubleshooting)  
20. [Project files](#project-files)  
21. [Safety](#license--safety-note)

---

## What it does (feature summary)

| Area | Feature |
|------|---------|
| **Pointer** | Move the OS cursor from the index fingertip; horizontal axis mirrored for natural “selfie” control. |
| **Click** | Open hand (all five fingers up) = left click. |
| **Scroll** | Thumb + index + middle up; move hand vertically to scroll (cursor mode). |
| **Cursor lock** | Pinky-only-up gesture freezes pointer and scroll until toggled again. |
| **Volume** | Pinch/spread thumb and index to set Windows master volume; optional lock with five fingers up. |
| **Brightness** | Same pinch metaphor for display brightness when WMI/API supports it (typical laptop panel). |
| **Typing** | Virtual QWERTY + numbers on the video overlay; pinch thumb–index on a key to type or click outside to focus fields. |
| **Drawing** | Mirrored strokes on a separate canvas blended over the camera; four colors; pinch-to-pick palette; eraser removes ink; clear with open hand. |
| **Navigation** | Hold index+middle (thumb spread) to cycle modes, or press `m`; fist hold or `q` to exit. |

---

## Requirements

| Item | Details |
|------|---------|
| **OS** | **Windows** — volume uses **pycaw** (Core Audio); brightness uses **screen-brightness-control** (often WMI). |
| **Python** | **3.10+** recommended; **3.13** is known to work with `requirements.txt`. |
| **Camera** | USB or built-in webcam; app opens device index **`0`** with **DirectShow** (`cv2.CAP_DSHOW`). |
| **Model file** | **`hand_landmarker.task`** in the project directory (same folder as `handtrackingModule.py`). Download or copy from the [MediaPipe Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) assets if missing. |
| **Lighting** | Good lighting improves landmark stability. |
| **Hands** | One hand is tracked (`maxHands=1`). |

---

## Hardware Components

| Component | Description |
|----------|------------|
| ESP32 Development Board | Microcontroller used to receive data and control LED display |
| MAX7219 8x8 LED Matrix (x2) | Displays typed characters |
| Jumper Wires | For connections |
| USB Cable | For ESP32 power and communication |
| Webcam | For hand gesture detection |

---

## Hardware Connections

Connect MAX7219 to ESP32 as follows:

| MAX7219 Pin | ESP32 Pin |
|------------|----------|
| VCC        | 5V       |
| GND        | GND      |
| DIN        | GPIO 23  |
| CS         | GPIO 5   |
| CLK        | GPIO 18  |

**Additional Notes:**
- Use 2 matrices connected in series (daisy-chain)
- Connect OUT of first matrix to IN of second
- Ensure proper power supply (5V recommended)

---

## ESP32 LED Display Integration

This project extends the virtual keyboard by sending typed characters to an ESP32, which displays them on MAX7219 LED matrices in real-time.

### Working Flow

Hand Gesture → Python (cursor.py) → Serial Communication → ESP32 → LED Matrix Display

### Key Behavior

| Input | Action on Display |
|------|------------------|
| Character | Displayed instantly |
| Backspace | Removes last character |
| Enter | Clears display |

### Communication Details

- Communication via USB Serial
- Baud rate: 115200
- Python sends characters directly
- ESP32 updates display instantly

---

## Installation

### Option A: Virtual environment (recommended)

```bash
cd AI_PROJECT
python -m venv env
.\env\Scripts\activate
pip install -r requirements.txt
```

### Option B: User / global install

```bash
pip install -r requirements.txt
```

Always run the app with the **same Python** that has the packages installed.

### Dependencies

| Package | Role |
|---------|------|
| `opencv-python` | Capture, drawing, window (`Virtual Mouse`), blending |
| `numpy` | Interpolation, clipping |
| `mediapipe` | Hand Landmarker **Tasks** API |
| `pyautogui` | Mouse move, click, scroll, key/character injection |
| `pycaw` | Windows default playback device volume |
| `comtypes` | Dependency of `pycaw` |
| `screen-brightness-control` | Optional at **import** time; if missing, brightness mode still runs but shows install hints |

---

## How to run

```bash
python cursor.py
```

The OpenCV window title is **`Virtual Mouse`**. It is created as **topmost** each frame so it tends to stay above other windows (you can still minimize it in the OS).

### Keyboard shortcuts (while the OpenCV window is focused)

| Key | Action |
|-----|--------|
| **`m`** | Advance to the **next** mode (same order as gesture cycling). |
| **`q`** | **Quit** the application immediately. |

### Exit via gesture

Hold a **full fist** (all fingers counted as down: `[0,0,0,0,0]`) for **`exitHoldSeconds` = 0.8 s**. A countdown **“Exit in … s”** appears.

---

## Running with ESP32 Display

1. Upload ESP32 code using Arduino IDE  
2. Connect ESP32 to your computer  
3. Close Arduino Serial Monitor  
4. Run the Python application:
   ```bash
   python cursor.py
   ```
5. Switch to keyboard mode  
6. Use pinch gesture to type  

Typed characters will appear instantly on the LED matrix.

---

## Finger notation

Hand states use five digits: **`[thumb, index, middle, ring, pinky]`**.

- **`1`** = that finger is **up / extended** (as classified by `handtrackingModule.fingersUp()`).  
- **`0`** = **down / folded**.

Example: **`[0, 1, 1, 0, 0]`** = only **index** and **middle** up (mode-switch pose).

---

## Modes and switching

### Mode order (fixed)

```
cursor → volume → brightness → keyboard → drawing → (wraps to cursor)
```

Internal list: `["cursor", "volume", "brightness", "keyboard", "drawing"]`.

### How to switch

1. **Gesture**  
   - Pose: **`[0, 1, 1, 0, 0]`** (index + middle up; thumb, ring, pinky down).  
   - Hold **~1.25 s** (`switchHoldSeconds`).  
   - **Thumb–index distance must be ≥ ~90 px** (`switchMinThumbIndexDist`). If the thumb is too close to the index, the timer **does not start**—this avoids accidental switches while pinching for volume/brightness.  
   - While waiting: **“Switch mode in … s”** is shown.

2. **Keyboard**  
   - Press **`m`** once per step.

After a successful gesture switch, **`wasPinching`** is set so an immediate open-hand **click** in cursor mode is suppressed once (debounce).

---

## Universal gestures (all modes)

| Action | How |
|--------|-----|
| **Next mode** | Index + middle up, thumb spread, hold **~1.25 s** — or **`m`** |
| **Quit** | Fist hold **~0.8 s** — or **`q`** |

The line *“Scroll: thumb+index+middle \| Switch: … \| Exit: fist”* is always drawn (scroll applies only in **cursor** mode).

---

## Cursor mode

Maps the camera working area to the full screen. Pointer coordinates are **smoothed** (`smoothening = 2.5`). Raw fingertip **X** is clamped to a margin **`frameR = 40`** from each vertical edge before mapping; **horizontal screen position** uses **`wScr - clocX`** (mirror).

| Gesture / condition | Action |
|---------------------|--------|
| **Index up** and **not** scroll pose and **not** locked | Move mouse if movement exceeds **`moveDeadZone` (2.0)** in smoothed space |
| **`[1,1,1,1,1]`** | **Left click** (one shot while hand stays open; **`wasPinching`** blocks repeat until fingers close) |
| **Thumb + index + middle up**; ring & pinky down | **Scroll**: vertical motion of **(index tip Y + middle tip Y) / 2** accumulates; when **\|accumulator\| ≥ 7** (`scrollThreshold`) and **> 0.015 s** since last scroll (`scrollCooldown`), emit scroll in steps of **±1…±3** × **90** (`scrollAmount`) |
| **`[0,0,0,0,1]`** (pinky only) | Toggle **cursor lock**: when locked, **no** move and **no** scroll |

**Status text:** `CURSOR: LIVE` / `CURSOR: LOCKED`.

**Note:** `pyautogui.FAILSAFE` is **False** and **`PAUSE` = 0** — corner failsafe and delays are off by design.

---

## Volume mode

Uses **`pycaw`** on the default **speakers** endpoint.

| Gesture / input | Action |
|-----------------|--------|
| **Thumb–index distance** | Maps **45–200 px** → volume scalar **0–1**; sent to **`SetMasterVolumeLevelScalar`** when change **> 0.01** and **≥ 0.03 s** since last API update |
| **`[1,1,1,1,1]`** | Toggles **volume lock** — when locked, distance no longer changes volume |

**UI:** Vertical bar ~`(50,150)–(85,400)`**, red fill, **percent** text; thumb–index link drawn in magenta.

**Status:** `VOL: LIVE` / `VOL: LOCKED`.

---

## Brightness mode

Uses **`screen_brightness_control`** if import succeeds; otherwise red on-screen install hint including **`sys.executable`**.

| Gesture / input | Action |
|-----------------|--------|
| **Thumb–index distance** | Maps **40–220 px** → **0–100%** brightness target; apply when value changes and **≥ 0.06 s** since last update |
| **`[1,1,1,1,1]`** | Toggles **brightness lock** |

On repeated **set** failures, **`brightnessApplyFailures`** increments; after **8** failures, **`brightnessGiveUp`** stops further attempts and an error message suggests laptop panel / unsupported external displays.

**UI:** Bar ~`(110,150)–(145,400)`**, yellow/cyan tint; **BRT: LIVE** / **BRT: LOCKED**.

---

## Keyboard mode

**Virtual keyboard** is drawn with keys from **`KEY_ROWS`**; **`key_at_position`** uses raw **`(ix, iy)`** (same space as overlay). The **OS cursor** is driven by smoothed, mirrored mapping like cursor mode (**`wScr - clocX`**).

| Gesture / input | Action |
|-----------------|--------|
| **Index tip over key** | Key highlighted **green** |
| **Pinch**: thumb–index distance **< `keyboardPinchThreshold` (32)** and **index up** | If **`keyPressCooldown` (0.25 s)** elapsed: type **`active_key`**, or **click** if tip is off keys |

**Key behavior**

| Key label | Effect |
|-----------|--------|
| `SPACE` | `pyautogui.press("space")` |
| `ENTER` | `pyautogui.press("enter")` |
| `BACK` | `pyautogui.press("backspace")` |
| `0-9`, `A-Z` | `pyautogui.write(key.lower())` |

**Layout (exact)**

```
1 2 3 4 5 6 7 8 9 0
Q W E R T Y U I O P
A S D F G H J K L
Z X C V B N M BACK
SPACE    ENTER
```

Geometry: **`KEY_W=50`**, **`KEY_H=45`**, **`KEY_GAP=6`**, first row starts at **`KEY_START_Y=240`**.

There is **no Shift**; letters are always sent **lowercase**.

---

## Drawing mode

### Rendering pipeline

1. Strokes are drawn into a **`uint8`** numpy canvas **`(hCam, wCam) = (480, 640)`**, same size as the camera frame.  
2. Each frame: **`img = cv2.addWeighted(img, 0.7, canvas, 1.0, 0.0)`** — camera weighted **0.7**, canvas **1.0** (per-channel blend; result is clipped).  
3. **Drawing UI** and **fingertip overlay** are drawn **after** this blend so they stay sharp and full brightness.

### Coordinates

- **`x1`** = index X clamped to **`[frameR, wCam - frameR]`**.  
- **`drawX = wCam - x1`** — **mirrored** drawing X (matches intuitive hand side).  
- **`pointer_y`**: **`iy`** when **`iy ≤ frameR`** (top band, palette), else **`y1`** (clamped vertical draw zone).  
- **Palette hit-testing** uses **`(drawX, iy)`** so the swatch under the **visible** fingertip matches selection.

### Gestures

| Gesture | Meaning |
|---------|---------|
| **Index up, middle down** | **Draw** (pen or eraser) |
| **Index + middle up** | **Hover** — no stroke; outer **green** ring on fingertip overlay |
| **`[1,1,1,1,1]`** | **Clear** entire canvas (latched once per open-hand gesture) |
| **`[0,0,1,0,0]`** | Toggle **eraser** (middle finger only up) |

### Brush thickness

Thumb–index distance **30–150 px** maps to:

| Tool | Thickness range |
|------|-----------------|
| **Pen** | **2–20 px** |
| **Eraser** | **15–35 px** |

Label **“Brush: N px”** is shown under the palette.

### Color palette

- **Four** colors (BGR in OpenCV): **red** `(0,0,255)`, **green** `(0,255,0)`, **blue** `(255,0,0)`, **yellow** `(0,255,255)`.  
- Swatches: **`DRAW_PALETTE_SW=56`**, **`SH=30`**, **`GAP=6`**, row at **`DRAW_PALETTE_TOP=34`**.  
- **Select:** index tip on swatch + **pinch** with thumb–index distance **< `DRAW_PALETTE_PINCH_THRESH` (48)** (looser than keyboard **32**), index up, **not** all-five-fingers clear pose.  
- **Preview** square: shows **`penColor`**; in eraser mode shows **gray + X**; **green border flash** ~**0.35 s** after a successful pick (`palette_pick_flash_until`).  
- **Hover:** swatch border **green** (ready) or **cyan** (still need to pinch); hints **“PINCH now…”** / **“On color — pinch…”**.

### Eraser (true removal)

Eraser strokes use **`(0, 0, 0)`** on the canvas — **no ink** in the blend layer — so previous colored strokes are **removed** along the path. (White would **add** light, not erase.)

### Fingertip overlay

Drawn **last**: white ring, black ring, filled core (cyan while drawing, magenta hover, orange/blue tints in eraser). **Not** multiplied by the 0.7 camera weight.

### Status line (drawing)

*Draw: index \| Hover: index+mid \| Clear: 5 \| Palette: pinch swatch \| Eraser: mid only*

---

## On-screen UI

| Element | Description |
|---------|-------------|
| **FPS** | Top-left, magenta |
| **Mode title** | e.g. `DRAWING MODE [fun]` |
| **Global hint** | Scroll (cursor), mode switch (~1.2 s, thumb spread), exit fist |
| **Mode hint** | Cursor lock / vol lock / brightness / keyboard pinch / drawing shortcuts |
| **Volume / brightness** | Bars + % + LIVE/LOCKED |
| **Brightness errors** | Pip hint + executable path, or API failure lines |
| **Drawing** | Dark panel, palette instructions, swatches, current color, brush px, ERASER label |

Window stays **topmost** via `cv2.WND_PROP_TOPMOST`.

---

## Technical details (implementation)

| Topic | Value / behavior |
|-------|------------------|
| **Resolution** | **`wCam=640`**, **`hCam=480`** |
| **Capture** | `CAP_PROP_FPS=60`, **MJPG** fourcc, **`CAP_PROP_BUFFERSIZE=1`** |
| **Hand detector** | `handDetector(maxHands=1)`, landmarks without drawing in the hot path |
| **Landmarks** | Index tip **8**, middle **12**, thumb **4** |
| **Smoothing** | Exponential-style: `cloc = ploc + (target - ploc) / smoothening` |
| **Brightness init** | If `sbc` loads, initial **`brightnessValue`** may sync from first **`get_brightness()`** read |
| **Optional overlay** | `drawOverlay` flag exists for a frame rectangle (off unless you enable it in code) |

---

## Troubleshooting

| Issue | What to try |
|-------|-------------|
| Camera busy / black | Close other apps; change `VideoCapture(0)` index if multiple cameras |
| Model error | Place **`hand_landmarker.task`** next to **`handtrackingModule.py`** |
| Import / module errors | `python -m pip install -r requirements.txt` with the **same** interpreter as **`python cursor.py`** |
| Venv points to removed Python | Recreate venv: `python -m venv env` then reinstall |
| Mode switch never counts down | **Spread thumb** from index (≥ ~90 px) while index+middle up |
| Brightness red text | Install **`screen-brightness-control`** for that Python |
| Brightness “API failed” | Use **internal laptop display**; many externals cannot be dimmed via WMI |
| Color pick unreliable | Use **mirrored** fingertip over swatch; pinch **firmly** (threshold **48** px) |
| Wrong monitor brightness | OS may bind brightness to one display only |

---

## Project files

| File | Role |
|------|------|
| **`cursor.py`** | Main application: modes, gestures, canvas, UI |
| **`handtrackingModule.py`** | MediaPipe Hand Landmarker, **`findHands`**, **`findPosition`**, **`fingersUp`** |
| **`requirements.txt`** | Pinned dependencies |
| **`hand_landmarker.task`** | Hand landmark model (required at runtime) |

---

## License / safety note

This software moves the **real** mouse cursor, **clicks**, **scrolls**, and can **type** into the focused application. It can change **system volume** and **screen brightness**. Only use it where that behavior is acceptable. Disable or close the app before leaving an unattended machine if others could trigger gestures on camera.
