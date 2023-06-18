import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import time

detector = HandDetector(maxHands=1)
folder = "D:\Personal\Sign Language Reader\ASL\Data\My Own\Test"
counter = 0
#
# img = cv2.imread("D:\Personal\Sign Language Reader\ASL\Data\Kaggle Data\\asl_dataset\\a\hand1_a_bot_seg_1_cropped.jpeg")
# hands, img = detector.findHands(img)
# # print(hands)
# if hands:
#     hand = hands[0]
#     x, y, w, h = hand['bbox']
#     x = img[y:y+h, x:x+w]
#     # frame = cv2.blur(img, ksize=(10, 10))
#     # print(frame.shape)
#     # print(x.shape)
#     # print(frame[y:y+h, x:x+w].shape)
#     # frame[y:y+h, x:x+w] = x
#     # cv2.imshow("Orig", img)
#     # cv2.imshow("Output", frame)
#     # cv2.waitKey(0)
#
#     blur = cv2.blur(x, ksize=(10, 10))
#     # img[y:y + h, x:x + w] = blur
#     cv2.imshow("Blr", blur)
#     cv2.waitKey(0)

cap = cv2.VideoCapture(0)
offset = 12

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    frame = img
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        a = img[y-offset:y+h+offset, x-offset:x+w+offset]
        frame = cv2.blur(img, ksize=(10, 10))
        frame[y-offset:y + h+offset, x-offset:x + w+offset] = a
    cv2.imshow("Output", frame)
    # cv2.imshow("Output", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}\Image_{time.time()}.jpg', frame)
        print(counter)

