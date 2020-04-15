import numpy as np
import pandas as pd
from PIL import Image 
import os
import cv2
from train import train_images as tr
from DetectImage import detect_image as detect

def train_recognizer():
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	tr_images = tr()
	images, image_id = tr_images.extractImages("./training_set") 
	image_id = np.array(image_id)
	recognizer.train(images, image_id)
	recognizer.save('trained.yml')
if not os.path.isfile('trained.yml'):
	train_recognizer()

d = detect()
d.identify('Record.xlsx', 'attendence.xlsx')


