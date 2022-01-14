import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import os
import random
from PIL import Image
import itertools


def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.YlOrBr):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = 5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


x_test = np.load('X_test.npy')
y_test = np.load('Y_test.npy')

sample_shape = x_test[0].shape
img_width, img_height = sample_shape[0], sample_shape[1]
input_shape = (img_width, img_height, 1)

input_test = x_test.reshape(len(x_test), input_shape[0], input_shape[1], input_shape[2])
x_test = input_test
x_test = (x_test / 255.0)

reconstructed_model = keras.models.load_model("my_model")
predictions = reconstructed_model.predict(x_test)
print(predictions)

predictions = tf.math.round(predictions)
predictions = tf.argmax(input=predictions, axis=1)

cm = tf.math.confusion_matrix(y_test, predictions)
cm = cm.numpy()
cm_plot_labels = ['cookie', 'smartphone', 'carrot', 'broccoli', 'floor lamp', 'grass', 'moon', 'mug', 'sword', 'sun']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')
