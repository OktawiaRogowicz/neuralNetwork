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
from tensorflow.keras.datasets import mnist

x_train = np.load('X_train.npy')
y_train = np.load('Y_train.npy')
x_test = np.load('X_test.npy')
y_test = np.load('Y_test.npy')

# (x_train, y_train), (x_test, y_test) = mnist.load_data()

sample_shape = x_train[0].shape
img_width, img_height = sample_shape[0], sample_shape[1]
input_shape = (img_width, img_height, 1)

input_train = x_train.reshape(len(x_train), input_shape[0], input_shape[1], input_shape[2])
input_test = x_test.reshape(len(x_test), input_shape[0], input_shape[1], input_shape[2])

x_train = input_train
x_test = input_test

x_train = (x_train / 255.0)
x_test = (x_test / 255.0)

model = Sequential([
    Convolution2D(filters=32, input_shape=(100, 100, 1), kernel_size=(3, 3), activation='relu', padding='valid'),
    BatchNormalization(),
    MaxPool2D((2, 2)),
    Convolution2D(filters=32, kernel_size=(3, 3), activation='relu', padding='valid'),
    BatchNormalization(),
    MaxPool2D((2, 2)),
    Convolution2D(filters=32, kernel_size=(3, 3), activation='relu', padding='valid'),
    BatchNormalization(),
    MaxPool2D((2, 2)),
    Convolution2D(filters=32, kernel_size=(3, 3), activation='relu', padding='valid'),
    BatchNormalization(),
    Flatten(),
    Dense(units=32, activation="relu"),
    # Dropout(0.15),
    Dense(units=16, activation="relu"),
    # Dropout(0.05),
    Dense(units=10, activation="softmax")
])

print(model.summary())

optim = RMSprop(learning_rate=0.0001)
model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(
    rotation_range=10,
    horizontal_flip=True,
    vertical_flip=False,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
)

x_train_long = np.repeat(a=x_train, repeats=10, axis=0)
y_train_long = np.repeat(a=y_train, repeats=10, axis=0)
x_test_long = np.repeat(a=x_test, repeats=10, axis=0)
y_test_long = np.repeat(a=y_test, repeats=10, axis=0)

x_train_long_length = len(x_train_long)

y_train_long = to_categorical(y_train_long)
y_test_long = to_categorical(y_test_long)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

batch_size = 64
train_generator = datagen.flow(x_train_long, y_train_long, batch_size=batch_size)

imgplot = plt.imshow(x_train[0])
plt.show()

figure = plt.figure()
i = 0

imX = x_train[0]
imY = np.asarray(['jakakolwiek-labelka'])
imX = np.expand_dims(imX, 0)

for x_batch, y_batch in datagen.flow(imX, imY):
   a = figure.add_subplot(5, 5, i + 1)
   plt.imshow(np.squeeze(x_batch))
   a.axis('off')
   if i == 24: break
   i += 1
figure.set_size_inches(np.array(figure.get_size_inches()) * 3)
plt.show()

datagen_valid = ImageDataGenerator(
    vertical_flip=False,
)

valid_steps = x_train.shape[0] // batch_size
validation_generator = datagen_valid.flow(x_train, y_train, batch_size=batch_size)

steps = x_train_long_length // batch_size

history = model.fit(
    train_generator,
    steps_per_epoch=x_train_long_length // batch_size,
    epochs=15,
    validation_data=validation_generator,
    validation_freq=1,
    validation_steps=valid_steps,
    verbose=2
)

model.save("my_model")

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
