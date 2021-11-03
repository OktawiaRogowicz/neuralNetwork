import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
from tensorflow.keras.utils import to_categorical

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import logging

matplotlib.use('TkAgg')


def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    print(x_train.shape, x_test.shape)

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),  # input layer
        keras.layers.Dense(128, activation='relu'),  # hidden layer
        keras.layers.Dense(10, activation='softmax')  # output layer
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x=x_train, y=y_train, epochs=6)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print(test_loss, test_acc)
    # history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test), verbose=0)
