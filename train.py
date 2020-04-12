import cv2
import os
from PIL import Image
import numpy as np



class train_images:

	def __init__(self):
				
		self.images = []
		self.imageId = []

	def extractImages(self, path):

		for root, dirs, files in os.walk(path):
			for fname in files:
				img_id = os.path.basename(root)
				img = os.path.join(root, fname)
				image = cv2.imread(img)

				gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
				face = cascade.detectMultiScale(gray_image, 1.1, 5)
				# print(img)
				# print(face)
				x, y, w, h = face[0]
				gray = gray_image[y:y+h, x:x+h]
				equalize = cv2.equalizeHist(gray)
				eq_image = cv2.medianBlur(equalize, 3)
				self.imageId.append(int(img_id))
				self.images.append(eq_image)

		return self.images, self.imageId



