import tensorflow as tf
# import pickle as fo
import joblib as fo
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.utils import normalize, to_categorical
from Models import PreTrainedModels

# ARRAYFOLDER = "Pickle File (RGB 224)"
# FILETYPE = "joblib"
CATEGORIES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]
print("Opening the pickle files=======================================================")
# X_train = fo.load(open("Pickle Files/X_train.pickle", "rb"))
# y_train = fo.load(open("Pickle Files/y_train.pickle", "rb"))
# X_test = fo.load(open("Pickle Files/X_test.pickle", "rb"))
# y_test = fo.load(open("Pickle Files/y_test.pickle", "rb"))

# X_train = fo.load(f"{ARRAYFOLDER}/X_train.{FILETYPE}")
# y_train = fo.load(f"{ARRAYFOLDER}/y_train.{FILETYPE}")
# X_test = fo.load(f"{ARRAYFOLDER}/X_test.{FILETYPE}")
# y_test = fo.load(f"{ARRAYFOLDER}/y_test.{FILETYPE}")

print("Calling to_categorical to convert y to binary segments")
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)


print("\n"*2)
print("Creating a Sequential model and adding layers (Conv2D, MaxPooling2D, Dense, FLatten)--------------------------------------->")

optimizer = 'adam'
# optimizer = 'rmsprop'

# activation = 'sigmoid'
activation = 'relu'

pretrained = PreTrainedModels()
mobilenet = pretrained.models(len(CATEGORIES))

# model = Sequential()
# model.add(Conv2D(32, 3, activation=activation, padding='same', input_shape= X_train.shape[1:]))
# model.add(BatchNormalization())
#
# model.add(Conv2D(32, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D())
#
# model.add(Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
# model.add(BatchNormalization())
#
# model.add(Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D())
#
# model.add(Flatten())
# model.add(Dense(128, activation = activation, kernel_initializer = 'he_uniform'))
# model.add(Dense(24, activation = 'softmax'))

model.compile(optimizer = optimizer,loss = 'categorical_crossentropy', metrics = ['accuracy'])
print("\n"*2)
print("="*20)
print(model.summary())
print("="*20)
print("\n"*2)

model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_test, y_test))
model.save('Model/model.h5')