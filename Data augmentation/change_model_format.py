import numpy as np
import tensorflow
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

import os
import random

from PIL import Image

model = keras.models.load_model("my_model")
model.save("model.h5")
