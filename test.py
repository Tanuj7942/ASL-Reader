import random

import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import os
import tensorflow as tf
import time
# import pyautogui

def preprocess_image(image_path):
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY)
    image = np.array(image).reshape(-1, 100, 100, 1)
    # image = cv2.resize(image, (-1, 100, 100, 1))
    image = image / 255.0
    # image = np.expand_dims(image, axis=0)
    return image

# model = tf.keras.models.load_model("Model/Models/Testing/A-Z (Except Q _ Z) (500!70)/keras_model.h5")

# cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Testing Models

classifier = Classifier("Model/Models/Testing/A-Z (Except J _ Z) (1000!70)/keras_model.h5", "Model/Models/Testing/A-Z (Except J _ Z) (1000!70)/labels.txt")
# classifier = Classifier("Models/Testing/A-C (500!70)/keras_model.h5", "Models/Testing/A-C (500!70)/labels.txt")
# classifier = Classifier("Models/Testing/A-C (1000!100)/keras_model.h5", "Models/Testing/A-C (1000!100)/labels.txt")
# classifier = Classifier("Model/Testing/A-Z (Except Q & Z) (1000!100)/keras_model.h5", "Models/Testing/A-Z (Except Q & Z) (1000!100)/labels.txt")

# Final Models

# Group Based Model
# classifier = Classifier("Models/Final/A-D (1000!70)/keras_model.h5", "Models/Final/A-D (1000!70)/labels.txt")
# classifier = Classifier("Models/Final/E-H (1000!70)/keras_model.h5", "Models/Final/E-H (1000!70)/labels.txt")
# classifier = Classifier("Models/Final/I-M (1000!70)/keras_model.h5", "Models/Final/I-M (1000!70)/labels.txt")
# classifier = Classifier("Models/Final/N-Q (1000!70)/keras_model.h5", "Models/Final/N-Q (1000!70)/labels.txt")
# classifier = Classifier("Models/Final/R-U (1000!70)/keras_model.h5", "Models/Final/R-U (1000!70)/labels.txt")
# classifier = Classifier("Models/Final/V-Y (1000!70)/keras_model.h5", "Models/Final/V-Y (1000!70)/labels.txt")

# Single Model

offset = 20
imgSize = 300
counter = 0

# labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]
labels = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y"]

KaggleData = "Kaggle Data\\asl_dataset"
KaggleData1 = "Kaggle Data 1\\asl_alphabet_train"
KaggleData2 = "Kaggle Data 2\\ASL_Dataset\\Train"

# while True:
#     try:
#         success, img = cap.read()
#         imgOutput = img.copy()
#         hands, img = detector.findHands(img)
#         if hands:
#             hand = hands[0]
#             x, y, w, h = hand['bbox']
#
#             imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
#             imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
#
#             imgCropShape = imgCrop.shape
#
#             aspectRatio = h/w
#
#             if aspectRatio > 1:
#                 k = imgSize/h
#                 wCal = math.ceil(k*w)
#                 imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#                 imgResizeShape = imgResize.shape
#                 wGap = math.ceil((imgSize-wCal)/2)
#                 imgWhite[:, wGap:wCal+wGap] = imgResize
#                 # imgWhite = preprocess_image(imgWhite)
#                 # prediction = model.predict(imgWhite)
#                 prediction, index = classifier.getPrediction(imgWhite, draw=False)
#                 # label = labels[int(prediction[0][0])]
#                 # print(prediction)
#                 print(prediction, index)
#
#             else:
#                 k = imgSize / w
#                 hCal = math.ceil(k * h)
#                 imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#                 imgResizeShape = imgResize.shape
#                 hGap = math.ceil((imgSize - hCal) / 2)
#                 imgWhite[hGap:hCal + hGap, :] = imgResize
#                 # imgWhite = preprocess_image(imgWhite)
#                 prediction, index = classifier.getPrediction(imgWhite, draw=False)
#                 print(prediction, index)
#                 # prediction = model.predict(imgWhite)
#                 # label = labels[int(prediction[0][0])]
#                 # print(prediction)
#
#         cv2.rectangle(imgOutput, (x-offset, y-offset-50), (x-offset+90, y-offset-50+50), (255, 0, 255), cv2.FILLED)
#         cv2.putText(imgOutput, labels[index], (x, y-26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
#         cv2.rectangle(imgOutput, (x - offset, y - offset), (x+w+offset, y+h+offset), (255, 0, 255), 4)
#     except Exception:
#         pass
#
#         # cv2.imshow("ImageCrop", imgCrop)
#         # cv2.imshow("ImageWhite", imgWhite)
#
#     cv2.imshow("Image", imgOutput)
#     key = cv2.waitKey(1)
#     if key == ord("s"):
#         pass
#         # screen_shot = pyautogui.screenshot()
#         # screen_shot.save("SS/screen_shot" + str(counter) + ".png")
#         # counter += 1

DIR = "D:\Personal\Sign Language Reader\ASL\Data"

for label in labels:
    # print(label)
    imgNum = random.randint(0, 5500)
    images = os.listdir(os.path.join(DIR, KaggleData2, label))
    imgPath = os.path.join(DIR, KaggleData2, label, images[imgNum])
    img = cv2.imread(imgPath)

    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    cv2.imwrite(f"Data/Random Images/Image_{time.time()}(Orig{label}).jpg", img)
    if hands:
        prediction, index = classifier.getPrediction(img, draw=False)
        counter += 1
        cv2.imwrite(f'Data/Test Outputs/Image_{time.time()}(Orig{label})({labels[index]}).jpg', img)
        print(label, imgNum, counter)