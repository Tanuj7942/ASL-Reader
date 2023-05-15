import tensorflow as tf
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.utils import normalize, to_categorical

print("Opening the pickle files=======================================================")
X_train = pickle.load(open("Pickle Files/X_train.pickle", "rb"))
y_train = pickle.load(open("Pickle Files/y_train.pickle", "rb"))
X_test = pickle.load(open("Pickle Files/X_test.pickle", "rb"))
y_test = pickle.load(open("Pickle Files/y_test.pickle", "rb"))

# print("Calling to_categorical to convert y to binary segments")
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)


print("\n"*2)
print("Creating a Sequential model and adding layers (Conv2D, MaxPooling2D, Dense, FLatten)--------------------------------------->")
activation = 'sigmoid'
model = Sequential()
model.add(Conv2D(32, 3, activation=activation, padding='same', input_shape= X_train.shape[1:]))
model.add(BatchNormalization())

model.add(Conv2D(32, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
model.add(BatchNormalization())

model.add(Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer = 'he_uniform'))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(128, activation = activation, kernel_initializer = 'he_uniform'))
model.add(Dense(24, activation = 'softmax'))

model.compile(optimizer = 'rmsprop',loss = 'categorical_crossentropy', metrics = ['accuracy'])
print("\n"*2)
print("="*20)
print(model.summary())
print("="*20)
print("\n"*2)

# model.fit(X_train, y_train, epochs=3, batch_size=32, validation_data=(X_test, y_test))