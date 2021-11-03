import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.layers import Convolution2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt

from quickdraw import QuickDrawDataGroup, QuickDrawData

qd = QuickDrawData()
print(qd.drawing_names)

names = ['cookie', 'crab', 'carrot', 'bat', 'floor lamp', 'grass', 'moon', 'mug', 'sword', 'sun']
expected_value = 0

x_train, y_train, x_test, y_test = [], [], [], []

for name in names:
    y = 0
    category = QuickDrawDataGroup(name, max_drawings=120)
    for drawing in category.drawings:
        data = np.asarray(drawing.get_image())
        # print(expected_value)
        if y < 100:
            x_train.append(data[:, :, 0])
            y_train.append(expected_value)
            # print("!")
        else:
            x_test.append(data[:, :, 0])
            y_test.append(expected_value)
        y = y + 1

    expected_value = expected_value + 1

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

x_train = (x_train / 255.0) - 0.5
x_test = (x_test / 255.0) - 0.5

x_train = tf.expand_dims(x_train, axis=-1)
x_test = tf.expand_dims(x_test, axis=-1)
y_train = tf.expand_dims(y_train, axis=-1)
y_test = tf.expand_dims(y_test, axis=-1)

model = Sequential([
    Convolution2D(filters=128, kernel_size=(5, 5), input_shape=(255, 255, 1), activation='relu', padding='same'),
    BatchNormalization(),
    Convolution2D(filters=128, kernel_size=(5, 5), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPool2D((2, 2)),
    Convolution2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same'),
    BatchNormalization(),
    Convolution2D(filters=64, kernel_size=(5, 5), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPool2D((2, 2)),
    Convolution2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same'),
    BatchNormalization(),
    Convolution2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPool2D((2, 2)),
    Convolution2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Convolution2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Flatten(),
    Dense(units=32, activation='relu'),
    Dropout(0.15),
    Dense(units=16, activation='relu'),
    Dropout(0.05),
    Dense(units=10, activation='softmax')
])

model.summary()

#optim = SGD(lr=0.001, momentum=0.5)
optim = RMSprop(lr=0.001)

#test

model.compile(optimizer = optim, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    x_train,
    to_categorical(y_train),
    epochs=80,
    validation_split=0.15,
    verbose=1
)

eval = model.evaluate(x_test, to_categorical(y_test))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
