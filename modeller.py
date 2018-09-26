import numpy as np
from keras.models import *
from keras.layers import *
from keras.utils import np_utils
from keras.callbacks import *
from matplotlib import pyplot as plt
from keras import backend as K
from keras.callbacks import *
from keras.optimizers import *
from keras.metrics import *
from keras.utils import *
import tensorflow as tf
import keras
import datetime

import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import itertools
import collections

def trainModel(model, num_epochs, filename, x_train, y_train, batch_size, sample_weight=None, class_weight=None, validation_data=None, validation_split=0.2):
	filepath = 'weights/' + filename
	checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_weights_only=True, save_best_only=True,
	                             mode='auto', period=1)
	tensor_board = TensorBoard(log_dir='logs/', histogram_freq=0, batch_size=batch_size)
	if sample_weight is not None:
		history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=num_epochs, verbose=1,
	                    callbacks=[checkpoint, tensor_board], validation_split=0.2, sample_weight = sample_weight)
	elif class_weight is not None:
		if validation_data is not None:
			history = model.fit(x_train, y_train, verbose=1, batch_size=batch_size, epochs=num_epochs,
			                    validation_data=validation_data,
			                    class_weight=class_weight,
			                    callbacks=[checkpoint, tensor_board])
		else:
			history = model.fit(x_train, y_train, verbose=1, batch_size=batch_size, epochs=num_epochs,
			                    validation_split=validation_split,
			                    class_weight=class_weight,
			                    callbacks=[checkpoint, tensor_board])

	return history


def evaluate_error(model, x_test, y_test):
	pred = model.predict(x_test, batch_size=32)
	pred = np.argmax(pred, axis=1)
	actual = np.argmax(y_test, axis=1)
	#pred = np.expand_dims(pred, axis=1)  # make same shape as y_test
	error = np.sum(np.not_equal(pred, actual)) / len(actual)

	return error