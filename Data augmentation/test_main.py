import numpy as np
import tensorflow
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

import os
import random

from PIL import Image

x_test = np.load('X_test.npy')
y_test = np.load('Y_test.npy')
first = x_test[0]
first_y = y_test[0]
print(x_test[0])

img = Image.fromarray(x_test[81], 'L')
img.show()

sample_shape = x_test[0].shape
img_width, img_height = sample_shape[0], sample_shape[1]
input_shape = (img_width, img_height, 1)

input_test = x_test.reshape(len(x_test), input_shape[0], input_shape[1], input_shape[2])

x_test = input_test

x_test = (x_test / 255.0)

x1 = x_test[0]
y1 = y_test[0]
y1 = to_categorical(y1)
reconstructed_model = keras.models.load_model("my_model")
predictions = reconstructed_model.predict(x_test)
print(predictions)
