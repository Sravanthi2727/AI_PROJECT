import cv2
import time
import numpy as np
import handtrackingModule as htm
import pyautogui
import math
from pycaw.pycaw import AudioUtilities

wCam, hCam = 640, 480
frameR = 40
smoothening = 2.5
moveDeadZone = 2.0

wasPinching = False
mode = "cursor"
modes = ["cursor", "volume"]
exitHoldSeconds = 0.8
exitGestureStart = None
switchHoldSeconds = 0.5
switchGestureStart = None
scrollThreshold = 7
scrollAmount = 90
scrollCooldown = 0.015
lastScrollY = None
lastScrollTime = 0.0
scrollAccumulator = 0.0

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

            # Scroll gesture: thumb + index + middle up -> [1,1,1,0,0]
            scrollGesture = (fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0 and fingers[4] == 0)
            if scrollGesture:
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

            if fingers[1] == 1 and not scrollGesture:
                if abs(clocX - plocX) > moveDeadZone or abs(clocY - plocY) > moveDeadZone:
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

    cTime = time.time()
    fps = 1/(cTime - pTime) if (cTime - pTime) != 0 else 0
    pTime = cTime

    modeText = "CURSOR MODE [cursor]"
    if mode == "volume":
        modeText = "VOLUME MODE [speaker]"
    cv2.putText(img, modeText, (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(img, "Scroll: thumb+index+middle | Switch: index+middle hold | Exit: fist hold", (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)
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