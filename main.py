import cv2
import numpy as np
import tensorflow as tf
import pickle

# Load the trained model
model = tf.keras.models.load_model("Model/model.h5")

# Load the one-hot encoded labels
y_train = pickle.load(open("Pickle Files/y_train.pickle", "rb"))
# y_test = pickle.load(open("Pickle Files/y_test.pickle", "rb"))

# Load and preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (100, 100))
    image = image / 255.0
    # image = np.expand_dims(image, axis=0)
    return image

# Path to the image you want to predict
image_path = "path/to/your/image.jpg"

# Preprocess the image
image = preprocess_image(image_path)

# Make predictions
predictions = model.predict(image)
predicted_class_index = np.argmax(predictions)
predicted_class_label = np.argmax(y_train[predicted_class_index])

# Print the predicted class label
print("Predicted class label:", predicted_class_label)
