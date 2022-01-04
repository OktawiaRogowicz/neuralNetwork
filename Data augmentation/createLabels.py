import os

from PIL import Image
import numpy as np
import re
import cv2

dirname = os.path.dirname(__file__)
input_dir = os.path.join(dirname, 'resized_images')

directories = os.listdir(input_dir)

result = []
labels = []

result_test = []
labels_test = []

for folder in directories:
	if os.path.isdir(input_dir + '/' + folder):
		folder2 = os.listdir(input_dir + '/' + folder)
		number = int(folder) - 1

		index = 0
		for image in folder2:
			im = Image.open(input_dir+"/"+folder+"/"+image).convert('L') #Opening image
			img = (np.array(im))
			out = np.array(img, np.uint8)

			if index < 70:
				result.append(out)
				labels.append(number)
			else:
				result_test.append(out)
				labels_test.append(number)
			index = index + 1

print(result)
np.save('X_train.npy', result)
np.save('Y_train.npy', labels)
np.save('X_test.npy', result_test)
np.save('Y_test.npy', labels_test)
