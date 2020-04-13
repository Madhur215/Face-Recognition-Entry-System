import numpy as np
import pandas as pd
import os
import cv2


names = { 0: 'niharika', 1 : 'madhur', 2: 'v', 3: 'l'}

class detect_image:

	def __init__(self):
		self.cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
		self.recognizer = cv2.face.LBPHFaceRecognizer_create()
		self.recognizer.read('trained.yml')

	def identify(self):
		cap = cv2.VideoCapture(0)
		f = 0
		while True:
			_, frame = cap.read()
			gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			equalize = cv2.equalizeHist(gray_frame)
			image = cv2.medianBlur(equalize, 3)
			face = self.cascade.detectMultiScale(image, 1.1, 5)

			for x, y, w, h in face:
				cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 4)
				label, conf = self.recognizer.predict(gray_frame[y:y+h, x:x+w])
				print(names.get(label))

				cv2.imshow("Detect", frame)
				if cv2.waitKey(0) and 0xFF == ord('q'):
					f = 1
					cap.release()
					break

			cv2.destroyAllWindows()

