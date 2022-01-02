import numpy as np
import tensorflow
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Convolution2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical

x_train = np.load('X_train.npy')
y_train = np.load('Y_train.npy')
x_test = np.load('X_test.npy')
y_test = np.load('Y_test.npy')

# Set input shape
sample_shape = x_train[0].shape
img_width, img_height = sample_shape[0], sample_shape[1]
input_shape = (img_width, img_height, 1)

# Reshape data
input_train = x_train.reshape(len(x_train), input_shape[0], input_shape[1], input_shape[2])
input_test = x_test.reshape(len(x_test), input_shape[0], input_shape[1], input_shape[2])

x_train = input_train
x_test = input_test

x_train = (x_train / 255.0) - 0.5
x_test = (x_test / 255.0) - 0.5

model = Sequential([
    Convolution2D(filters=128, kernel_size=(5, 5), input_shape=(100, 100, 1), activation='relu', padding='same'),
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
    Dense(units=32, activation="relu"),
    Dropout(0.15),
    Dense(units=16, activation="relu"),
    Dropout(0.05),
    Dense(units=10, activation="softmax")
])
optim = RMSprop(learning_rate=0.001)
model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

# datagen = ImageDataGenerator(
#     rotation_range=10,
#     horizontal_flip=True,
#     vertical_flip=False,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     rescale=1. / 255,
#     shear_range=0.05,
#     zoom_range=0.05,
# )
#
# x_train_length = len(x_train)
#
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
#
# batch_size = 64
# train_generator = datagen.flow(x_train, y_train, batch_size=batch_size)
#
# datagen_valid = ImageDataGenerator(
#     rescale=1. / 255,
# )
#
# x_valid = x_train[:100 * batch_size]
# y_valid = y_train[:100 * batch_size]
#
# valid_steps = x_valid.shape[0] // batch_size
# validation_generator = datagen_valid.flow(x_valid, y_valid, batch_size=batch_size)
#
# steps = x_train_length // batch_size
#
# history = model.fit(
#     train_generator,
#     steps_per_epoch=x_train_length // batch_size,
#     epochs=120,
#     validation_data=validation_generator,
#     validation_freq=1,
#     validation_steps=valid_steps,
#     verbose=2
# )

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

history = model.fit(
   x_train,
   y_train,
   epochs=80,
   validation_split=0.15,
   verbose=2
)

eval = model.evaluate(x_test, y_test)
print(eval)

print(history.history.keys())

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.show()
