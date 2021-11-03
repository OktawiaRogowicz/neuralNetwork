import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import logging

matplotlib.use('TkAgg')

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    print(train_images.shape)

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),   # input layer
        keras.layers.Dense(128, activation='relu'),   # hidden layer
        keras.layers.Dense(10, activation='softmax')  # output layer
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, batch_size=100, epochs=10)
    print("done")

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)

    print('Test accuracy:', test_acc)
