import cv2
# from cvzone.HandTrackingModule import HandDetector
import mediapipe as mp
import os

DATADIR = "Data"
CollectedDir = "My Own"
KaggleData = "Kaggle Data\\asl_dataset"
KaggleData1 = "Kaggle Data1\\asl_alphabet_train"
kaggleData2 = "Kaggle Data2\\ASL_Dataset\\Train"

# CATEGORIES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]
CATEGORIES = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y"]
# CATEGORIES = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]

mpDraw = mp.solutions.drawing_utils
mpHands = mp.solutions.hands


# detector = HandDetector(maxHands=1, detectionCon=0.3)

# def create_training_data():
# for category in CATEGORIES:
#     path = os.path.join(DATADIR, KaggleData, category)
#     # class_num = CATEGORIES.index(category)
#     for img in os.listdir(path):
#         try:
#             img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
#             print(type (img_array))
#             hands, img = detector.findHands(img)
#             print(hands, img)
#             if hands:
#                 cv2.imwrite("Image", img)
#             break
#             # new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
#             # training_data.append([new_array, class_num])
#         except Exception as e:
#             print(e)
#     break


img = cv2.imread(f"{DATADIR}/{KaggleData}/a/hand1_a_bot_seg_3_cropped.jpeg")
# img = cv2.imread("D:/Personal/Sign Language Reader/ASL/Data/Kaggle Data/asl_dataset/a/hand1_a_bot_seg_1_cropped.jpeg")

# cap = cv2.VideoCapture(0)
# while True:
#     success, img = cap.read()
# hands, img = detector.findHands(img)
# cv2.imshow("Image", img)
# cv2.waitKey(10000)
# cv2.destroyAllWindows()

with mp_hands.Hands(min_detection_confidence=0.8,
                    min_tracking_confidence=0.5) as hands:  # You can pass `max_num_hands` argument here as well if you want to detect more that one hand
    image = cv2.imread(f"{DATADIR}/{KaggleData}/a/hand1_a_bot_seg_3_cropped.jpeg")
    image = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)

    image.flags.writeable = False

    results = hands.process(image)

    image.flags.writeable = True

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    print(results)

    # Rendering results
    if results.multi_hand_landmarks:
        for num, hand in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS,
                                      )

    cv2.imwrite(os.path.join('Output Images', '{}.jpg'.format(uuid.uuid1())), image)
    cv2_imshow(image)

cv2.destroyAllWindows()