# import the necessary packages
# from pyimagesearch.gradcam import GradCAM
from GradCAM import GradCAM
import tensorflow
# from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.preprocessing.image import load_img
# from tensorflow.keras.applications import imagenet_utils
import numpy as np
# import argparse
import imutils
import cv2


CATEGORIES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]

IMG_SIZE = 100
# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to the input image")
# ap.add_argument("-m", "--model", type=str, default="vgg",
# 	choices=("vgg", "resnet"),
# 	help="model to be used")
# args = vars(ap.parse_args())

# initialize the model to be VGG16
# Model = VGG16
# check to see if we are using ResNet
# if args["model"] == "resnet":
# 	Model = ResNet50
# load the pre-trained CNN from disk
print("[INFO] loading model...")
# model = Model(weights="imagenet")
model = tensorflow.keras.models.load_model("../Model/Models/Testing/A-Z (Except Q _ Z) (500!70)/keras_model.h5")

# load the original image from disk (in OpenCV format) and then
# resize the image to its target dimensions
# orig = cv2.imread(args["image"])
orig = cv2.imread("D:\Personal\Sign Language Reader\ASL\Data\My Own\Test\Image_1685080138.4997108.jpg", cv2.IMREAD_GRAYSCALE)
image = orig.copy()
# orig = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))
# orig = orig / 255.0
# orig = np.expand_dims(orig, axis=0)

# load the input image from disk (in Keras/TensorFlow format) and
# preprocess it
# image = load_img("D:\Personal\Sign Language Reader\ASL\Data\My Own\Test\Image_1684817109.7892542.jpg", target_size=(IMG_SIZE, IMG_SIZE))
# image = img_to_array(image)
image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
image = image / 255.0
image = np.expand_dims(image, axis=0)
# image = imagenet_utils.preprocess_input(image)

# use the network to make predictions on the input image and find
# the class label index with the largest corresponding probability
preds = model.predict(image)
i = np.argmax(preds[0])
print(CATEGORIES[int(preds[0][0])])
# decode the ImageNet predictions to obtain the human-readable label
# decoded = imagenet_utils.decode_predictions(preds)
# (imagenetID, label, prob) = decoded[0][0]
# label = "{}: {:.2f}%".format(label, prob * 100)
# print("[INFO] {}".format(label))

# initialize our gradient class activation map and build the heatmap
cam = GradCAM(model, i)
heatmap = cam.compute_heatmap(image)
# resize the resulting heatmap to the original input image dimensions
# and then overlay heatmap on top of the image
heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
(heatmap, output) = cam.overlay_heatmap(heatmap, orig, alpha=0.5)
# heatmap = cam.overlay_heatmap(heatmap, orig, alpha=0.5)

# draw the predicted label on the output image
# cv2.rectangle(output, (0, 0), (340, 40), (0, 0, 0), -1)
# cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
# 	0.8, (255, 255, 255), 2)
# display the original image and resulting heatmap and output image
# to our screen
output = np.vstack([cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB), heatmap, output])
output = imutils.resize(output, height=700)
# heatmap = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2RGB)
# cv2.imshow("Output", orig)
# cv2.imshow("heatmap", heatmap)

# cv2.imshow("Output", output)
cv2.imwrite(f"D:\Personal\Sign Language Reader\ASL\Data\Test Outputs\output7({CATEGORIES[int(preds[0][0])]}).jpg", output)
cv2.waitKey(0)