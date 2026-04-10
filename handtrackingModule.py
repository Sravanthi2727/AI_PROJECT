import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers import landmark as landmark_module
import time
import math
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

# Finger tip and pip landmark indices
TIP_IDS = [4, 8, 12, 16, 20]
PIP_IDS = [3, 6, 10, 14, 18]

BaseOptions = mp_python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        # mode=False means continuous/video processing (faster for live webcam)
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self._results = None
        self._img = None
        self.running_mode = VisionRunningMode.IMAGE if mode else VisionRunningMode.VIDEO

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=self.running_mode,
            num_hands=maxHands,
            min_hand_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon,
        )
        self.detector = HandLandmarker.create_from_options(options)

    def findHands(self, img, draw=True):
        self._img = img
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self.running_mode == VisionRunningMode.VIDEO:
            timestamp_ms = time.monotonic_ns() // 1_000_000
            self._results = self.detector.detect_for_video(mp_image, timestamp_ms)
        else:
            self._results = self.detector.detect(mp_image)

        if draw and self._results.hand_landmarks:
            h, w, _ = img.shape
            for hand_landmarks in self._results.hand_landmarks:
                # draw connections
                connections = [
                    (0,1),(1,2),(2,3),(3,4),
                    (0,5),(5,6),(6,7),(7,8),
                    (5,9),(9,10),(10,11),(11,12),
                    (9,13),(13,14),(14,15),(15,16),
                    (13,17),(17,18),(18,19),(19,20),(0,17)
                ]
                pts = [(int(lm.x*w), int(lm.y*h)) for lm in hand_landmarks]
                for a, b in connections:
                    cv2.line(img, pts[a], pts[b], (0, 255, 0), 2)
                for pt in pts:
                    cv2.circle(img, pt, 5, (255, 0, 255), cv2.FILLED)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if not self._results or not self._results.hand_landmarks:
            return lmList
        if handNo >= len(self._results.hand_landmarks):
            return lmList

        h, w, _ = img.shape
        for id, lm in enumerate(self._results.hand_landmarks[handNo]):
            cx, cy = int(lm.x * w), int(lm.y * h)
            lmList.append([id, cx, cy])
            if draw:
                cv2.circle(img, (cx, cy), 7, (255, 0, 255), cv2.FILLED)
        return lmList

    def fingersUp(self):
        if not self._results or not self._results.hand_landmarks:
            return [0, 0, 0, 0, 0]

        h, w, _ = self._img.shape
        lmList = []
        for lm in self._results.hand_landmarks[0]:
            lmList.append([int(lm.x * w), int(lm.y * h)])

        fingers = []
        # Thumb: compare x positions
        if lmList[TIP_IDS[0]][0] > lmList[TIP_IDS[0] - 1][0]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other 4 fingers: tip y < pip y means finger is up
        for i in range(1, 5):
            if lmList[TIP_IDS[i]][1] < lmList[PIP_IDS[i]][1]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    detector = handDetector()
    pTime = 0

    while True:
        success, img = cap.read()
        if not success:
            continue

        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        if lmList:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) != 0 else 0
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
