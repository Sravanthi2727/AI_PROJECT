import cv2
import time
import numpy as np
import handtrackingModule as htm
import pyautogui
import math
from pycaw.pycaw import AudioUtilities

try:
    import screen_brightness_control as sbc
except ImportError:
    sbc = None

wCam, hCam = 640, 480
frameR = 40
smoothening = 2.5
moveDeadZone = 2.0

wasPinching = False
mode = "cursor"
modes = ["cursor", "volume", "brightness", "keyboard", "drawing"]
exitHoldSeconds = 0.8
exitGestureStart = None
switchHoldSeconds = 1.25
# Ignore mode-switch gesture while thumb and index are close (same pose as volume/brightness pinch).
switchMinThumbIndexDist = 90
switchGestureStart = None
scrollThreshold = 7
scrollAmount = 90
scrollCooldown = 0.015
lastScrollY = None
lastScrollTime = 0.0
scrollAccumulator = 0.0
keyboardPinchThreshold = 32
keyPressCooldown = 0.25
lastKeyPressTime = 0.0

KEY_ROWS = [
    list("1234567890"),
    list("QWERTYUIOP"),
    list("ASDFGHJKL"),
    ["Z", "X", "C", "V", "B", "N", "M", "BACK"],
    ["SPACE", "ENTER"],
]
KEY_W, KEY_H = 50, 45
KEY_GAP = 6
KEY_START_X, KEY_START_Y = 15, 240

# Drawing mode setup.
canvas = np.zeros((hCam, wCam, 3), dtype=np.uint8)
drawColor = (0, 255, 255)
brushThickness = 6
drawPrevX, drawPrevY = None, None
clearLatch = False
cursorLocked = False
cursorToggleLatch = False

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, wCam)
cap.set(4, hCam)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

detector = htm.handDetector(maxHands=1)

wScr, hScr = pyautogui.size()

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
pyautogui.MINIMUM_DURATION = 0

cv2.namedWindow("Virtual Mouse", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Virtual Mouse", cv2.WND_PROP_TOPMOST, 1)

drawOverlay = False

# System volume setup.
device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume
minVol, maxVol, _ = volume.GetVolumeRange()
lastVol = volume.GetMasterVolumeLevel()
lastVolScalar = volume.GetMasterVolumeLevelScalar()
lastVolUpdate = 0.0
volumeLocked = False
volumeToggleLatch = False
brightnessLocked = False
brightnessToggleLatch = False
# Stops trying after repeated set failures (e.g. no WMI/DDC display). Never conflate with "pip missing".
brightnessGiveUp = False
brightnessValue = 50
lastBrightnessUpdate = 0.0
brightnessApplyFailures = 0


def _read_system_brightness():
    """Return first valid brightness reading, or None if unavailable."""
    if sbc is None:
        return None
    try:
        for v in sbc.get_brightness():
            if v is not None:
                return int(v)
    except Exception:
        pass
    for idx in range(4):
        try:
            vals = sbc.get_brightness(display=idx)
            for v in vals or []:
                if v is not None:
                    return int(v)
        except Exception:
            continue
    return None


def _set_system_brightness(level):
    """Apply brightness; prefer all controllable displays, then display=0. Returns True on success."""
    if sbc is None:
        return False
    level = int(max(0, min(100, level)))
    try:
        sbc.set_brightness(level)
        return True
    except Exception:
        pass
    try:
        sbc.set_brightness(level, display=0)
        return True
    except Exception:
        return False


if sbc is not None:
    got = _read_system_brightness()
    if got is not None:
        brightnessValue = got


def draw_keyboard(img, active_key=None):
    for row_idx, row in enumerate(KEY_ROWS):
        y = KEY_START_Y + row_idx * (KEY_H + KEY_GAP)
        row_width = len(row) * KEY_W + (len(row) - 1) * KEY_GAP
        x = (wCam - row_width) // 2
        for key in row:
            w = KEY_W
            if key == "SPACE":
                w = KEY_W * 4
            elif key in ("BACK", "ENTER"):
                w = KEY_W * 2
            color = (60, 60, 60) if key != active_key else (0, 200, 0)
            cv2.rectangle(img, (x, y), (x + w, y + KEY_H), color, cv2.FILLED)
            cv2.rectangle(img, (x, y), (x + w, y + KEY_H), (255, 255, 255), 1)
            font_scale = 0.55 if key in ("SPACE", "BACK", "ENTER") else 0.65
            text_w = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0]
            text_x = x + (w - text_w) // 2
            cv2.putText(img, key, (text_x, y + 29),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
            x += w + KEY_GAP


def key_at_position(x_pos, y_pos):
    for row_idx, row in enumerate(KEY_ROWS):
        y = KEY_START_Y + row_idx * (KEY_H + KEY_GAP)
        row_width = len(row) * KEY_W + (len(row) - 1) * KEY_GAP
        x = (wCam - row_width) // 2
        for key in row:
            w = KEY_W
            if key == "SPACE":
                w = KEY_W * 4
            elif key in ("BACK", "ENTER"):
                w = KEY_W * 2
            if x <= x_pos <= x + w and y <= y_pos <= y + KEY_H:
                return key
            x += w + KEY_GAP
    return None

while True:
    success, img = cap.read()
    if not success:
        continue

    # Disable heavy landmark drawing in hot path for lower latency.
    img = detector.findHands(img, draw=False)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        ix, iy = lmList[8][1], lmList[8][2]  # Index tip
        mx, my = lmList[12][1], lmList[12][2]  # Middle tip
        tx, ty = lmList[4][1], lmList[4][2]  # Thumb tip
        thumb_index_dist = math.hypot(tx - ix, ty - iy)
        fingers = detector.fingersUp()
        now = time.time()

        # Exit gesture: close all 5 fingers (fist) for a short hold.
        if fingers == [0, 0, 0, 0, 0]:
            if exitGestureStart is None:
                exitGestureStart = now
            holdLeft = max(0.0, exitHoldSeconds - (now - exitGestureStart))
            cv2.putText(img, f'Exit in {holdLeft:.1f}s', (20, 115),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if now - exitGestureStart >= exitHoldSeconds:
                break
        else:
            exitGestureStart = None

        # Mode switch gesture (hold): index + middle up -> [0,1,1,0,0]
        # Require thumb far from index so this does not fire during brightness/volume pinch.
        if fingers == [0, 1, 1, 0, 0] and thumb_index_dist >= switchMinThumbIndexDist:
            if switchGestureStart is None:
                switchGestureStart = now
            switchLeft = max(0.0, switchHoldSeconds - (now - switchGestureStart))
            cv2.putText(img, f'Switch mode in {switchLeft:.1f}s', (20, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            if now - switchGestureStart >= switchHoldSeconds:
                mode = modes[(modes.index(mode) + 1) % len(modes)]
                switchGestureStart = None
                wasPinching = True  # debounce accidental click right after mode change
        else:
            switchGestureStart = None

        if drawOverlay:
            cv2.rectangle(img, (frameR, frameR), (wCam-frameR, hCam-frameR),
                          (255, 0, 255), 2)

        if mode == "cursor":
            x1 = min(max(ix, frameR), wCam - frameR)
            y1 = min(max(iy, frameR), hCam - frameR)
            x3 = (x1 - frameR) * wScr / (wCam - 2 * frameR)
            y3 = (y1 - frameR) * hScr / (hCam - 2 * frameR)

            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # Cursor lock gesture: pinky-only up -> [0,0,0,0,1]
            if fingers == [0, 0, 0, 0, 1]:
                if not cursorToggleLatch:
                    cursorLocked = not cursorLocked
                    cursorToggleLatch = True
            else:
                cursorToggleLatch = False

            # Scroll gesture: thumb + index + middle up -> [1,1,1,0,0]
            scrollGesture = (fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0)
            if scrollGesture and not cursorLocked:
                currentScrollY = (iy + my) / 2.0
                if lastScrollY is not None:
                    # Up movement => positive accumulator => scroll up
                    scrollAccumulator += (lastScrollY - currentScrollY)

                if (abs(scrollAccumulator) >= scrollThreshold) and ((now - lastScrollTime) > scrollCooldown):
                    steps = int(scrollAccumulator / scrollThreshold)
                    # Limit burst to keep behavior smooth.
                    steps = max(-3, min(3, steps))
                    if steps != 0:
                        pyautogui.scroll(steps * scrollAmount)
                        scrollAccumulator -= steps * scrollThreshold
                        lastScrollTime = now

                lastScrollY = currentScrollY
            else:
                lastScrollY = None
                scrollAccumulator = 0.0

            if fingers[1] == 1 and not scrollGesture and not cursorLocked:
                if abs(clocX - plocX) > moveDeadZone or abs(clocY - plocY) > moveDeadZone:
                    pyautogui.moveTo(wScr - clocX, clocY)

            plocX, plocY = clocX, clocY

            # Click gesture: all 5 fingers up (single click per gesture hold).
            if fingers == [1, 1, 1, 1, 1]:
                if not wasPinching and not cursorLocked:
                    pyautogui.click()
                    wasPinching = True
                    cv2.circle(img, (ix, iy), 12, (0, 255, 0), cv2.FILLED)
            else:
                wasPinching = False

            cv2.circle(img, (ix, iy), 8, (255, 0, 255), cv2.FILLED)
            lockText = "CURSOR: LOCKED" if cursorLocked else "CURSOR: LIVE"
            lockColor = (0, 255, 0) if cursorLocked else (0, 200, 255)
            cv2.putText(img, lockText, (20, 175),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, lockColor, 2)
        elif mode == "volume":
            # Volume mode: control master volume with thumb-index distance.
            lastScrollY = None
            scrollAccumulator = 0.0
            # 5 fingers up toggles lock/unlock for stable volume hold.
            if fingers == [1, 1, 1, 1, 1]:
                if not volumeToggleLatch:
                    volumeLocked = not volumeLocked
                    volumeToggleLatch = True
            else:
                volumeToggleLatch = False

            if not volumeLocked:
                length = math.hypot(tx - ix, ty - iy)
                volScalar = float(np.interp(length, [45, 200], [0.0, 1.0]))
                volScalar = max(0.0, min(1.0, volScalar))

                # Throttle expensive volume API calls to keep loop fast.
                if (abs(volScalar - lastVolScalar) > 0.01) and (now - lastVolUpdate > 0.03):
                    volume.SetMasterVolumeLevelScalar(volScalar, None)
                    lastVolScalar = volScalar
                    # Keep dB value synced for internal consistency if needed elsewhere.
                    lastVol = np.interp(lastVolScalar, [0.0, 1.0], [minVol, maxVol])
                    lastVolUpdate = now

            # Display from scalar to match Windows system volume percentage.
            volBar = int(np.interp(lastVolScalar, [0.0, 1.0], [400, 150]))
            volPer = int(lastVolScalar * 100)
            cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 2)
            cv2.rectangle(img, (50, volBar), (85, 400), (255, 0, 0), cv2.FILLED)
            cv2.putText(img, f'{volPer} %', (40, 440),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            lockText = "VOL: LOCKED" if volumeLocked else "VOL: LIVE"
            lockColor = (0, 255, 0) if volumeLocked else (0, 200, 255)
            cv2.putText(img, lockText, (35, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, lockColor, 2)
            cv2.circle(img, (tx, ty), 8, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (ix, iy), 8, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (tx, ty), (ix, iy), (255, 0, 255), 2)
        elif mode == "brightness":
            # Brightness mode: control screen brightness with thumb-index distance.
            lastScrollY = None
            scrollAccumulator = 0.0

            # 5 fingers up toggles lock/unlock for stable brightness hold.
            if fingers == [1, 1, 1, 1, 1]:
                if not brightnessToggleLatch:
                    brightnessLocked = not brightnessLocked
                    brightnessToggleLatch = True
            else:
                brightnessToggleLatch = False

            if sbc is not None and not brightnessGiveUp and not brightnessLocked:
                length = thumb_index_dist
                brightnessTarget = int(np.interp(length, [40, 220], [0, 100]))
                brightnessTarget = max(0, min(100, brightnessTarget))

                if (abs(brightnessTarget - brightnessValue) > 0) and (now - lastBrightnessUpdate > 0.06):
                    if _set_system_brightness(brightnessTarget):
                        brightnessApplyFailures = 0
                        confirmed = _read_system_brightness()
                        brightnessValue = confirmed if confirmed is not None else brightnessTarget
                    else:
                        brightnessApplyFailures += 1
                        if brightnessApplyFailures >= 8:
                            brightnessGiveUp = True
                    lastBrightnessUpdate = now

            briBar = int(np.interp(brightnessValue, [0, 100], [400, 150]))
            cv2.rectangle(img, (110, 150), (145, 400), (0, 220, 255), 2)
            cv2.rectangle(img, (110, briBar), (145, 400), (0, 220, 255), cv2.FILLED)
            cv2.putText(img, f'{brightnessValue} %', (100, 440),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 220, 255), 2)
            lockText = "BRT: LOCKED" if brightnessLocked else "BRT: LIVE"
            lockColor = (0, 255, 0) if brightnessLocked else (0, 200, 255)
            cv2.putText(img, lockText, (95, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, lockColor, 2)
            if sbc is None:
                cv2.putText(img, "pip install screen-brightness-control", (180, 205),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            elif brightnessGiveUp:
                cv2.putText(img, "Brightness API failed - external monitors often unsupported", (20, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2)
                cv2.putText(img, "cannot be dimmed; try the laptop screen", (20, 222),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2)

            cv2.circle(img, (tx, ty), 8, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (ix, iy), 8, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (tx, ty), (ix, iy), (255, 0, 255), 2)
        elif mode == "keyboard":
            # Keyboard mode: move cursor with index and pinch to type.
            lastScrollY = None
            scrollAccumulator = 0.0
            x1 = min(max(ix, frameR), wCam - frameR)
            y1 = min(max(iy, frameR), hCam - frameR)
            x3 = (x1 - frameR) * wScr / (wCam - 2 * frameR)
            y3 = (y1 - frameR) * hScr / (hCam - 2 * frameR)
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            if abs(clocX - plocX) > moveDeadZone or abs(clocY - plocY) > moveDeadZone:
                pyautogui.moveTo(wScr - clocX, clocY)
            plocX, plocY = clocX, clocY

            active_key = key_at_position(ix, iy)
            draw_keyboard(img, active_key)
            cv2.circle(img, (ix, iy), 8, (255, 0, 255), cv2.FILLED)

            pinchLen = math.hypot(tx - ix, ty - iy)
            pinchPress = pinchLen < keyboardPinchThreshold and fingers[1] == 1
            if pinchPress and (now - lastKeyPressTime > keyPressCooldown):
                if active_key == "SPACE":
                    pyautogui.press("space")
                elif active_key == "BACK":
                    pyautogui.press("backspace")
                elif active_key == "ENTER":
                    pyautogui.press("enter")
                elif active_key:
                    pyautogui.write(active_key.lower())
                else:
                    # Pinch outside keyboard acts as regular click (focus input box).
                    pyautogui.click()
                lastKeyPressTime = now
        else:
            # Drawing mode: index up draws, index+middle up moves without drawing.
            lastScrollY = None
            scrollAccumulator = 0.0
            x1 = min(max(ix, frameR), wCam - frameR)
            y1 = min(max(iy, frameR), hCam - frameR)

            # Draw state gestures.
            drawGesture = (fingers[1] == 1 and fingers[2] == 0)
            hoverGesture = (fingers[1] == 1 and fingers[2] == 1)

            if drawGesture:
                if drawPrevX is None:
                    drawPrevX, drawPrevY = x1, y1
                cv2.line(canvas, (drawPrevX, drawPrevY), (x1, y1), drawColor, brushThickness)
                drawPrevX, drawPrevY = x1, y1
            else:
                drawPrevX, drawPrevY = None, None

            # Clear canvas on 5-finger gesture (latching avoids repeated clears).
            if fingers == [1, 1, 1, 1, 1]:
                if not clearLatch:
                    canvas[:] = 0
                    clearLatch = True
            else:
                clearLatch = False

            cursorColor = (0, 255, 255) if drawGesture else (255, 0, 255)
            cv2.circle(img, (x1, y1), 8, cursorColor, cv2.FILLED)
            if hoverGesture:
                cv2.circle(img, (x1, y1), 16, (0, 255, 0), 1)

    # Blend drawing canvas on top of camera frame.
    if mode == "drawing":
        img = cv2.addWeighted(img, 0.7, canvas, 1.0, 0.0)

    cTime = time.time()
    fps = 1/(cTime - pTime) if (cTime - pTime) != 0 else 0
    pTime = cTime

    modeText = "CURSOR MODE [cursor]"
    if mode == "volume":
        modeText = "VOLUME MODE [speaker]"
    elif mode == "brightness":
        modeText = "BRIGHTNESS MODE [sun]"
    elif mode == "keyboard":
        modeText = "KEYBOARD MODE [typing]"
    elif mode == "drawing":
        modeText = "DRAWING MODE [fun]"
    cv2.putText(img, modeText, (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(img, "Scroll: thumb+index+middle | Switch: index+middle hold ~1.2s (thumb spread) | Exit: fist", (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 255, 200), 1)
    if mode == "cursor":
        cv2.putText(img, "Cursor lock: pinky only (toggle)", (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1)
    elif mode == "volume":
        cv2.putText(img, "Volume lock: 5 fingers (toggle)", (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1)
    elif mode == "brightness":
        cv2.putText(img, "Brightness: thumb-index distance | Lock: 5 fingers", (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1)
    elif mode == "keyboard":
        cv2.putText(img, "Keyboard: pinch on key to type", (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1)
    elif mode == "drawing":
        cv2.putText(img, "Draw: index up | Hover: index+middle | Clear: 5 fingers", (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1)
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50),
                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow("Virtual Mouse", img)
    # Keep window visible/topmost; user can still minimize/maximize manually.
    cv2.setWindowProperty("Virtual Mouse", cv2.WND_PROP_TOPMOST, 1)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('m'):
        mode = modes[(modes.index(mode) + 1) % len(modes)]
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()