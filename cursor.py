import cv2
import time
import numpy as np
import handtrackingModule as htm
import pyautogui

wCam, hCam = 640, 480
frameR = 100
smoothening = 7

pinchThreshold = 35
wasPinching = False

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

detector = htm.handDetector(maxHands=1)

wScr, hScr = pyautogui.size()

pyautogui.FAILSAFE = False

cv2.namedWindow("Virtual Mouse", cv2.WINDOW_NORMAL)

paused = False

while True:
    success, img = cap.read()
    if not success:
        continue

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1], lmList[8][2]
        x2, y2 = lmList[4][1], lmList[4][2]

        cv2.rectangle(img, (frameR, frameR), (wCam-frameR, hCam-frameR),
                      (255, 0, 255), 2)

        x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
        y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))

        clocX = plocX + (x3 - plocX) / smoothening
        clocY = plocY + (y3 - plocY) / smoothening

        fingers = detector.fingersUp()

        if fingers == [1,1,1,1,1]:
            paused = True
        else:
            paused = False

        if fingers[1] == 1:
            pyautogui.moveTo(wScr - clocX, clocY)     

        if fingers[1] == 1 and not paused:
            pyautogui.moveTo(wScr - clocX, clocY)

        plocX, plocY = clocX, clocY

        length = np.hypot(x2 - x1, y2 - y1)

        if length < pinchThreshold and not paused:
            if not wasPinching:
                pyautogui.click()
                wasPinching = True
                cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
        else:
            wasPinching = False

        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)

    cTime = time.time()
    fps = 1/(cTime - pTime) if (cTime - pTime) != 0 else 0
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 50),
                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow("Virtual Mouse", img)
    cv2.setWindowProperty("Virtual Mouse", cv2.WND_PROP_TOPMOST, 1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()