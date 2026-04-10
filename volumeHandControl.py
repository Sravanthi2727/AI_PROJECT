import cv2
import time
import numpy as np
import handtrackingModule as htm
import math

from pycaw.pycaw import AudioUtilities

########################
wCam, hCam = 640, 480
########################

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0


detector = htm.handDetector(detectionCon=0.7)


device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume

minVol, maxVol, _ = volume.GetVolumeRange()

print("Volume Range:", minVol, maxVol)

########################

while True:
    success, img = cap.read()
    if not success:
        continue

    
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

       
        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)


        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

        
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

        
        length = math.hypot(x2 - x1, y2 - y1)

        
        vol = np.interp(length, [50, 200], [minVol, maxVol])
        volume.SetMasterVolumeLevel(vol, None)

        
        if length < 50:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

        
        volBar = np.interp(length, [50, 200], [400, 150])
        volPer = np.interp(length, [50, 200], [0, 100])

        cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 2)
        cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, f'{int(volPer)} %', (40, 450),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

   
    cTime = time.time()
    fps = 1/(cTime - pTime) if (cTime - pTime) != 0 else 0
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (10, 70),
                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

    cv2.imshow("Volume Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()