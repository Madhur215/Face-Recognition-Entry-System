import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.layers import MaxPooling2D
from keras.layers import Convolution2D
from keras.layers import Flatten
from keras.models import Sequential

class CNN:

	def __init__(self):

		self.model = Sequential()
		self.model.add(Convolution2D(32, 3, 3, input_shape=(128, 128, 3), activation='relu'))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Convolution2D(32, 3, 3, activation='relu'))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))

		self.model.add(Flatten())
		self.model.add(Dense(output_dim=64, activation='relu'))
		self.model.add(Dense(output_dim=2, activation='softmax'))

		self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

	def fit_model(self, x_train, y_train):
		self.model.fit(x_train, y_train, epochs=50)

	def predict(self, x_test):
		return self.model.predict(x_test)

