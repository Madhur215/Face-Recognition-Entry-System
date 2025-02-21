import numpy as np
import pandas as pd
from PIL import Image 
import os
import cv2
from train import train_images as tr
from DetectImage import detect_image as detect
import tkinter as tk

def train_recognizer():
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	tr_images = tr()
	images, image_id = tr_images.extractImages("./training_set") 
	image_id = np.array(image_id)
	recognizer.train(images, image_id)
	recognizer.save('trained.yml')

if not os.path.isfile('trained.yml'):
		train_recognizer()

root = tk.Tk()
root.geometry('300x200')

root.title('FARES')
frame = tk.Frame(root, bg="#FFF")
frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)
d = detect('Record.xlsx', 'attendence.xlsx')

def mark_entry():
	p_name = d.identify()
	cv2.destroyAllWindows()
	print(p_name)
	d.markEntry(p_name)

def close_entry():
	p_name = d.identify()
	cv2.destroyAllWindows()
	print(p_name)
	d.close_entry(p_name)

mark_entry_button = tk.Button(root, text="Mark Entry", command=mark_entry)
mark_entry_button.place(relx=0.2, rely=0.5)
close_entry_button = tk.Button(root, text="close Entry",command = close_entry)
close_entry_button.place(relx=0.6, rely=0.5)

if __name__ == '__main__':
	root.mainloop()


