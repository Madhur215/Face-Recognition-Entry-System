import numpy as np
import pandas as pd
import os
import cv2
import openpyxl
from openpyxl import load_workbook


names = { 0: 'niharika', 1 : 'madhur', 2: 'v', 3: 'l'}

class detect_image:

	def __init__(self):
		self.cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
		self.recognizer = cv2.face.LBPHFaceRecognizer_create()
		self.recognizer.read('trained.yml')
		self.students_marked = []

	def identify(self, recordSheet, attendenceSheet):
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
				# print(names.get(label))
				p_name = names.get(label)
				self.markEntry(p_name, recordSheet, attendenceSheet)

				cv2.imshow("Recognizing Face", frame)
				flag = cv2.waitKey(1) & 0xFF
				f += 1
				if f > 20:
					cap.release()
					break	

			
	def markEntry(self, p_name, recordSheet, attendenceSheet):
		if p_name in attendenceSheet:
			print("Entry already marked!")
		else:
			self.students_marked.append(p_name)
			recordWorkbook = load_workbook(filename=recordSheet)
			sheet = recordWorkbook.active
			recordNames = sheet["A"]
			print(recordNames)








