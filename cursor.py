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

wasPinching = False
mode = "cursor"
exitHoldSeconds = 0.8
exitGestureStart = None
switchHoldSeconds = 0.5
switchGestureStart = None

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
                mode = "volume" if mode == "cursor" else "cursor"
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

            # Click gesture: all 5 fingers up (single click per gesture hold).
            if fingers == [1, 1, 1, 1, 1]:
                if not wasPinching:
                    pyautogui.click()
                    wasPinching = True
                    cv2.circle(img, (ix, iy), 12, (0, 255, 0), cv2.FILLED)
            else:
                wasPinching = False

            cv2.circle(img, (ix, iy), 8, (255, 0, 255), cv2.FILLED)
        else:
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

    cTime = time.time()
    fps = 1/(cTime - pTime) if (cTime - pTime) != 0 else 0
    pTime = cTime

    modeText = "CURSOR MODE [cursor]"
    if mode == "volume":
        modeText = "VOLUME MODE [speaker]"
    cv2.putText(img, modeText, (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(img, "Switch: index+middle hold | Exit: fist hold", (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50),
                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow("Virtual Mouse", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('m'):
        mode = "volume" if mode == "cursor" else "cursor"
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()