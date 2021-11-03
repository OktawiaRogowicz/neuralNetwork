import numpy as np

import tensorflow
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.layers import Convolution2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = (x_train / 255.0) - 0.5
x_test = (x_test / 255.0) - 0.5

model = Sequential([
    Convolution2D(filters=128, kernel_size=(5, 5), input_shape=(32, 32, 3), activation='relu'),
    BatchNormalization(),
    Convolution2D(filters=128, kernel_size=(5, 5), activation='relu'),
    BatchNormalization(),
    MaxPool2D((2, 2)),
    Convolution2D(filters=64, kernel_size=(5, 5), activation='relu'),
    BatchNormalization(),
    Convolution2D(filters=64, kernel_size=(5, 5), activation='relu'),
    BatchNormalization(),
    MaxPool2D((2, 2)),
    Convolution2D(filters=32, kernel_size=(5, 5), activation='relu'),
    BatchNormalization(),
    Convolution2D(filters=32, kernel_size=(5, 5), activation='relu'),
    BatchNormalization(),
    MaxPool2D((2, 2)),
    Convolution2D(filters=16, kernel_size=(3, 3), activation='relu'),
    BatchNormalization(),
    Convolution2D(filters=16, kernel_size=(3, 3), activation='relu'),
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

model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

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

