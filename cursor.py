import cv2
import time
import numpy as np
import handtrackingModule as htm
import pyautogui
import math
from pycaw.pycaw import AudioUtilities

wCam, hCam = 640, 480
frameR = 40
smoothening = 3

pinchThreshold = 35
wasPinching = False
mode = "cursor"
modes = ["cursor", "volume", "keyboard"]
exitHoldSeconds = 0.8
exitGestureStart = None
switchHoldSeconds = 0.5
switchGestureStart = None
keyPressCooldown = 0.35
lastKeyPressTime = 0.0
shiftOn = False

KEY_ROWS = [
    ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"],
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L"],
    ["SHIFT", "Z", "X", "C", "V", "B", "N", "M", "BACK"],
    ["SPACE", "ENTER"],
]
KEY_W, KEY_H = 50, 52
KEY_GAP = 6
KEY_START_X, KEY_START_Y = 25, 150

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

drawOverlay = False

# System volume setup.
device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume
minVol, maxVol, _ = volume.GetVolumeRange()
lastVol = volume.GetMasterVolumeLevel()
lastVolUpdate = 0.0


def draw_keyboard(img, active_key=None):
    for row_idx, row in enumerate(KEY_ROWS):
        y = KEY_START_Y + row_idx * (KEY_H + KEY_GAP)
        row_width = len(row) * KEY_W + (len(row) - 1) * KEY_GAP
        x = KEY_START_X + (wCam - 2 * KEY_START_X - row_width) // 2

        for key in row:
            w = KEY_W
            if key in ("SHIFT", "BACK", "ENTER"):
                w = KEY_W * 2
            elif key == "SPACE":
                w = KEY_W * 4
            color = (60, 60, 60) if key != active_key else (0, 200, 0)
            cv2.rectangle(img, (x, y), (x + w, y + KEY_H), color, cv2.FILLED)
            cv2.rectangle(img, (x, y), (x + w, y + KEY_H), (255, 255, 255), 1)

            label = key
            font_scale = 0.65 if key not in ("SPACE", "BACK", "ENTER") else 0.5
            text_w = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0][0]
            text_x = x + (w - text_w) // 2
            text_y = y + 30
            cv2.putText(img, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
            x += w + KEY_GAP


def key_at_position(x_pos, y_pos):
    for row_idx, row in enumerate(KEY_ROWS):
        y = KEY_START_Y + row_idx * (KEY_H + KEY_GAP)
        row_width = len(row) * KEY_W + (len(row) - 1) * KEY_GAP
        x = KEY_START_X + (wCam - 2 * KEY_START_X - row_width) // 2
        for key in row:
            w = KEY_W
            if key in ("SHIFT", "BACK", "ENTER"):
                w = KEY_W * 2
            elif key == "SPACE":
                w = KEY_W * 4
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
        tx, ty = lmList[4][1], lmList[4][2]  # Thumb tip
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

        # Dedicated mode-switch gesture to avoid conflict with click/volume gestures.
        # Switch gesture: index + middle up -> [0,1,1,0,0]
        if fingers == [0, 1, 1, 0, 0]:
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

            if fingers[1] == 1:
                pyautogui.moveTo(wScr - clocX, clocY)

            plocX, plocY = clocX, clocY

            # Click gesture: thumb + index pinch (faster than 5-finger gesture).
            pinchLen = math.hypot(tx - ix, ty - iy)
            if pinchLen < pinchThreshold and fingers[1] == 1:
                if not wasPinching:
                    pyautogui.click()
                    wasPinching = True
                    cv2.circle(img, (ix, iy), 12, (0, 255, 0), cv2.FILLED)
            else:
                wasPinching = False

            cv2.circle(img, (ix, iy), 8, (255, 0, 255), cv2.FILLED)
        elif mode == "volume":
            # Volume mode: control master volume with thumb-index distance.
            length = math.hypot(tx - ix, ty - iy)
            vol = np.interp(length, [45, 200], [minVol, maxVol])
            vol = max(minVol, min(maxVol, vol))

            # Throttle expensive volume API calls to keep loop fast.
            if (abs(vol - lastVol) > 1.0) and (now - lastVolUpdate > 0.03):
                volume.SetMasterVolumeLevel(vol, None)
                lastVol = vol
                lastVolUpdate = now

            volBar = int(np.interp(vol, [minVol, maxVol], [400, 150]))
            volPer = int(np.interp(vol, [minVol, maxVol], [0, 100]))
            cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 2)
            cv2.rectangle(img, (50, volBar), (85, 400), (255, 0, 0), cv2.FILLED)
            cv2.putText(img, f'{volPer} %', (40, 440),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.circle(img, (tx, ty), 8, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (ix, iy), 8, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (tx, ty), (ix, iy), (255, 0, 255), 2)
        else:
            # In keyboard mode, still move OS cursor so user can focus any text box.
            x1 = min(max(ix, frameR), wCam - frameR)
            y1 = min(max(iy, frameR), hCam - frameR)
            x3 = (x1 - frameR) * wScr / (wCam - 2 * frameR)
            y3 = (y1 - frameR) * hScr / (hCam - 2 * frameR)
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening
            pyautogui.moveTo(wScr - clocX, clocY)
            plocX, plocY = clocX, clocY

            active_key = key_at_position(ix, iy)
            draw_keyboard(img, active_key)
            cv2.circle(img, (ix, iy), 8, (255, 0, 255), cv2.FILLED)
            pinchLen = math.hypot(tx - ix, ty - iy)

            # Keyboard press gesture: thumb + index pinch.
            if active_key and pinchLen < pinchThreshold and fingers[1] == 1 and (now - lastKeyPressTime > keyPressCooldown):
                if active_key == "SPACE":
                    pyautogui.press("space")
                elif active_key == "BACK":
                    pyautogui.press("backspace")
                elif active_key == "ENTER":
                    pyautogui.press("enter")
                elif active_key == "SHIFT":
                    shiftOn = not shiftOn
                else:
                    key_to_type = active_key
                    if active_key.isalpha():
                        key_to_type = active_key.upper() if shiftOn else active_key.lower()
                    pyautogui.write(key_to_type)
                    # One-shot shift behavior for easier typing.
                    if shiftOn:
                        shiftOn = False
                lastKeyPressTime = now
            # If no key is highlighted, pinch acts as normal mouse click
            # to place cursor focus in search bars / text boxes.
            elif (not active_key) and pinchLen < pinchThreshold and fingers[1] == 1 and (now - lastKeyPressTime > keyPressCooldown):
                pyautogui.click()
                lastKeyPressTime = now

    cTime = time.time()
    fps = 1/(cTime - pTime) if (cTime - pTime) != 0 else 0
    pTime = cTime

    modeText = "CURSOR MODE [cursor]"
    if mode == "volume":
        modeText = "VOLUME MODE [speaker]"
    elif mode == "keyboard":
        modeText = "KEYBOARD MODE [typing]"
    cv2.putText(img, modeText, (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(img, "Switch: index+middle hold | Exit: fist hold", (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)
    if mode == "keyboard":
        shift_text = "SHIFT: ON" if shiftOn else "SHIFT: OFF"
        cv2.putText(img, shift_text, (500, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50),
                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow("Virtual Mouse", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('m'):
        mode = modes[(modes.index(mode) + 1) % len(modes)]
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()