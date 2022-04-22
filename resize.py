import os

from PIL import Image
import numpy as np
import re
import cv2

dirname = os.path.dirname(__file__)
input_dir = os.path.join(dirname, 'images')

directories = os.listdir(input_dir)

for folder in directories:
	folder2 = os.listdir(input_dir + '/' + folder)
	number = re.findall(r'\d+', folder)
	number = number[0]

	for image in folder2:
		im = Image.open(input_dir+"/"+folder+"/"+image).convert('RGBA')
		background = Image.new('RGBA', im.size, (255, 255, 255))
		alpha_composite = Image.alpha_composite(background, im)
		resized_image = alpha_composite.resize((100, 100))

		path = "resized_images/" + number
		isExist = os.path.exists(path)
		if not isExist:
			os.makedirs(path)
		resized_image.save("resized_images/" + number + "/" + image, optimize=True, quality=100)