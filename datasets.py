import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import pickle
import random
from sklearn.model_selection import train_test_split

DATADIR = "Data"
CATEGORIES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]
# CATEGORIES = ["A", "B"]

training_data = []
IMG_SIZE = 150

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

print("Creating Training Data ============================================================================")
print("\n"*2)
create_training_data()
print("Performing a shuffle")
random.shuffle(training_data)

X = []
y = []

print("Creating a list as X(features) and y(labels)")
for features, label in training_data:
  X.append(features)
  y.append(label)

print("Converting to numpy array")
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
# X = np.array(X)
y = np.array(y)

print("Printing the shape of original data")
print("X----->  ",X.shape)
print("y----->  ",y.shape)

print("Spliting the data to training and testing data in 80/20 %")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size = .8)

print("Normalizing the X_train and X_test model by dividing the array with 255.0")
X_train = X_train/255.0
X_test = X_test/255.0

print("Saving all the test and training data in a pickle file seperately")
pickle_out = open("Pickle Files/X_train.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("Pickle Files/X_test.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_out = open("Pickle Files/y_train.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("Pickle Files/y_test.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

print("\n"*2)
print("Completed loading the dataset =======================================================================")
print("\n"*2)

print("Printing the shape of the data \n")
print("X_train----->  ",X_train.shape)
print("X_test----->  ",X_test.shape)
print("y_train----->  ",y_train.shape)
print("y_test----->  ",y_test.shape)

