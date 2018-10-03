'''Train a simple deep CNN on the CIFAR10 small images dataset.
It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from keras.datasets import mnist, cifar10
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

import os

import modeller as mod


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


def quantifyConfusion(model, xTestData, yTestData, comparisonIndicies=None, indexNames=None):
	pred = model.predict(xTestData)
	pred = np.argmax(pred, axis=1)
	yTestData = np.argmax(yTestData, axis=1)
	cm = confusion_matrix(yTestData, pred)
	# plot_confusion_matrix(cm,classes=['0','1'], title='All')

	A_fpr = cm[0][1] / np.sum(cm[0])
	A_fnr = cm[1][0] / np.sum(cm[1])

	print("--------------------------------------------")
	print("Overall FPR: %.5f" % A_fpr)
	print("Overall FNR: %.5f" % A_fnr)
	print("--------------------------------------------")
	if comparisonIndicies is not None:
		print("Index\tFPR\t\t\tFNR\t\t\tUnfairness")
		fprs = []
		fnrs = []
		for index in comparisonIndicies:
			cm = confusion_matrix(yTestData[xTestData[:, index] == 1], pred[xTestData[:, index] == 1], labels = [0, 1])
			fpr = cm[0][1] / np.sum(cm[0])
			fnr = cm[1][0] / np.sum(cm[1])
			fprs.append(fpr)
			fnrs.append(fnr)

		for i in range(len(fprs)):
			fpr = fprs[i]
			fnr = fnrs[i]
			avg = (sum(fprs) - fpr)/(len(fprs) - 1)
			unfairness = avg-fpr
			if indexNames is None:
				print("%i\t\t%.2f%%\t\t%.2f%%\t\t%.2f%%" % (comparisonIndicies[i], fpr*100, fnr*100, unfairness*100))
			else:
				print("%s\t\t%.2f%%\t\t%.2f%%\t\t%.2f%%" % (indexNames[i][:6], fpr*100, fnr*100, unfairness*100))

		print("--------------------------------------------")
	# plt.figure(figsize=(10,4))
	# cm = confusion_matrix(yTestData[xTestData[:, comparison1Index] == 1], pred[xTestData[:, comparison1Index] == 1])
	# # plt.subplot(1,2,1)
	# # plot_confusion_matrix(cm,classes=['0','1'], title='Female')
	#
	# FPR1 = cm[0][1] / np.sum(cm[0])
	# FNR1 = cm[1][0] / np.sum(cm[1])
	# discrimination1First = np.sum(cm[:, 1]) / np.sum(cm)
	# discrimination1Second = np.sum(cm[:, 0]) / np.sum(cm)
	#
	# # plt.subplot(1,2,2)
	# cm = confusion_matrix(yTestData[xTestData[:, comparison2Index] == 1], pred[xTestData[:, comparison2Index] == 1])
	# # plot_confusion_matrix(cm,classes=['0','1'], title='Male')
	#
	# M_fpr = cm[0][1] / np.sum(cm[0])
	# M_fnr = cm[1][0] / np.sum(cm[1])
	# M_dis = np.sum(cm[:, 1]) / np.sum(cm)
	# M_dis2 = np.sum(cm[:, 0]) / np.sum(cm)
	#
	# falsePositiveRate = np.abs(FPR1 - M_fpr)
	# falseNegativeRate = np.abs(FNR1 - M_fnr)
	#
	# unfairness = falsePositiveRate
	#
	# discrimination = np.abs(M_dis - discrimination1First)
	# discrimination2 = np.abs(M_dis2 - discrimination1Second)
	#
	# #print("%05d iterations..." % (k + 1), normal, sam_weight)
	# #print(model.evaluate(x_test, to_categorical(y_test), verbose=0))
	# print("FPR_all/M/F/W/B: %.5f, %.5f, %.5f" % (A_fpr, M_fpr, FPR1))
	# print("FNR_all/M/F/W/B: %.5f, %.5f, %.5f" % (A_fnr, M_fnr, FNR1))
	# print("Unfairness: %.5f" % unfairness)
	# print("Discrimination: %.5f" % discrimination)
	# print("Discrimination2: %.5f" % discrimination2)
	# print("\n")
	# for index in comparisonIndicies:
	# 	cm = confusion_matrix(yTestData[xTestData[:, index] == 1], pred[xTestData[:, index] == 1])
	# 	fprs.append(cm[0][1] / np.sum(cm[0]))
	# 	fnrs.append(cm[1][0] / np.sum(cm[1]))


def sampling32(args):
	z_mean, z_log_var = args
	n_dim = 32
	epsilon = K.random_normal(shape=(K.shape(z_mean)[0], n_dim), mean=0.,stddev=1.0)
	sample = z_mean + K.exp(z_log_var/2.0) * epsilon
	return sample


def sampling64(args):
	z_mean, z_log_var = args
	n_dim = 64
	epsilon = K.random_normal(shape=(K.shape(z_mean)[0], n_dim), mean=0.,stddev=1.0)
	sample = z_mean + K.exp(z_log_var/2.0) * epsilon
	return sample


def getBaseModel(modelInput):
	x = Reshape((21, 20, 1))(modelInput)
	x = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Conv2D(16, (3, 3), padding='same', activation='relu', strides=(3, 2))(x)
	x = Dropout(0.5)(x)
	x = Conv2D(16, (3, 3), padding='same', activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Conv2D(16, (3, 3), padding='same', activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Conv2D(32, (3, 3), padding='same', activation='relu', strides=(1, 2))(x)
	x = Dropout(0.5)(x)
	x = Flatten()(x)
	x = Dense(64, activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Dense(32, activation='relu')(x)
	modelOutput = Dense(2, activation='softmax')(x)
	model = Model(modelInput, modelOutput)
	model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['acc'])
	return model


def getAutoencoderModel(modelInput):
	x = Reshape((21, 20, 1))(modelInput)
	x = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Conv2D(16, (3, 3), padding='same', activation='relu', strides=(3, 2))(x)
	x = Dropout(0.5)(x)
	x = Conv2D(16, (3, 3), padding='same', activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Conv2D(16, (3, 3), padding='same', activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Conv2D(32, (3, 3), padding='same', activation='relu', strides=(1, 2))(x)
	x = Dropout(0.5)(x)
	x = Flatten()(x)
	x = Dense(64, activation='relu')(x)
	x = Dropout(0.5)(x)
	z_mean = Dense(32)(x)
	z_log_var = Dense(32)(x)
	x = Lambda(sampling32)([z_mean, z_log_var])
	x = Dense(5 * 7 * 32, activation='relu')(x)
	x = Reshape((7, 5, 32))(x)
	x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Conv2DTranspose(16, (3, 3), padding='same', activation='relu', strides=(1, 2))(x)
	x = Dropout(0.5)(x)
	x = Conv2D(16, (3, 3), padding='same', activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Conv2D(16, (3, 3), padding='same', activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Conv2DTranspose(8, (3, 3), padding='same', activation='relu', strides=(3, 2))(x)
	x = Dropout(0.5)(x)
	x = Conv2D(1, (3, 3), padding='same', activation='relu')(x)
	dec_out = Flatten()(x)

	encoder = Model(modelInput, z_mean)
	model = Model(modelInput, dec_out)
	model.compile(loss=autoencoder_loss(z_log_var, z_mean), optimizer=Adam(), metrics=['acc'])
	return model


def autoencoder_loss(z_log_var, z_mean, y_inp, alpha=1.0, normal=False, n_dim=1, n_class=1):
	def f_(x, x_decoded):
		if normal:
			kl_loss = - 0.5 * K.mean(1 + z_log_var -
			                         K.square(z_mean) - K.exp(z_log_var), axis=-1)
		else:
			label = tf.reshape(tf.tile(tf.reshape(y_inp, (-1, 1)), (1, n_dim // n_class)), (-1, n_dim))
			kl_loss = - 0.5 * K.mean(1 + z_log_var -
			                         K.square(z_mean - label) - K.exp(z_log_var), axis=-1)
		xent_loss = K.mean(K.binary_crossentropy(K.flatten(x), K.flatten(x_decoded)), axis=-1)
		return xent_loss + kl_loss * alpha

	return f_


def getM1Model(modelInput):
	n_dim = 64
	n_class = 2

	y_inp = Input((n_class,))
	x = Lambda(mod.padInput(11))(modelInput)
	x = Reshape((21, 20, 1))(x)
	x = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Conv2D(16, (3, 3), padding='same', activation='relu', strides=(3, 2))(x)
	x = Dropout(0.5)(x)
	x = Conv2D(16, (3, 3), padding='same', activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Conv2D(16, (3, 3), padding='same', activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Conv2D(32, (3, 3), padding='same', activation='relu', strides=(1, 2))(x)
	x = Dropout(0.5)(x)
	x = Flatten()(x)
	z_mean = Dense(n_dim)(x)
	z_log_var = Dense(n_dim)(x)
	auxilarily_out = Dense(n_class, activation='softmax', name='gen')(z_mean)

	# enc_out = BatchNormalization()(enc_out)
	x = Lambda(sampling64)([z_mean, z_log_var])
	x = Dense(5 * 7 * 32, activation='relu')(x)
	x = Reshape((7, 5, 32))(x)
	x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Conv2DTranspose(16, (3, 3), padding='same', activation='relu', strides=(1, 2))(x)
	x = Dropout(0.5)(x)
	x = Conv2D(16, (3, 3), padding='same', activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Conv2D(16, (3, 3), padding='same', activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Conv2DTranspose(8, (3, 3), padding='same', activation='relu', strides=(3, 2))(x)
	x = Dropout(0.5)(x)
	x = Conv2D(1, (3, 3), padding='same', activation='relu')(x)
	x = Flatten(name='recon')(x)
	dec_out = Lambda(mod.sliceOutput(409))(x)

	encoder = Model(modelInput, z_mean, name='encoder')
	encoder.trainable = False

	autoencoder = Model([modelInput, y_inp], [dec_out, auxilarily_out])
	autoencoder.compile(optimizer='adam', loss=[autoencoder_loss(z_log_var, z_mean, y_inp, n_dim=n_dim, n_class=n_class), 'categorical_crossentropy'], metrics=['acc'])
	return autoencoder, encoder


def getM2Model(modelInput, protectedIndices):
	x = Dense(64, activation="relu")(modelInput)
	x = Dropout(0.5)(x)
	x = Dense(64, activation="relu")(x)
	x = Dropout(0.5)(x)
	x = Dense(32, activation="relu")(x)
	x = BatchNormalization()(x)
	x = Dropout(0.5)(x)
	x = Dense(32, activation="relu")(x)
	x = BatchNormalization()(x)
	x = Dropout(0.5)(x)
	x = Dense(32, activation="relu")(x)
	lc = Dense(2, activation="softmax")(x)

	#indicies_protected = list(columns.index(x) for x in ["sex==Female", "sex==Male"])
	#indices_protected = list([femaleColumnIndex, maleColumnIndex])
	ll = Lambda(concat_input(protectedIndices))([lc, modelInput])

	model = Model(modelInput, lc)
	model.compile(loss=model2loss(1.5), optimizer=Adam(), metrics=[model2Accuracy])
	trainer = Model(modelInput, ll)
	trainer.compile(loss=model2loss(1.5), optimizer=Adam(), metrics=[model2Accuracy])
	return trainer, model


def concat_input(indicies):
	def fn(layers):
		layer_prev, layer_input = layers
		return tf.concat([layer_prev, tf.gather(layer_input, indicies, axis=1)], axis=1)
	return fn


def stackModels(modelInput, modelA, modelB):
	outputA = modelA(modelInput)
	outputB = modelB(outputA)
	return Model(modelInput, outputB)


def model2Accuracy(y_true, ll):
	return categorical_accuracy(y_true, ll[:, :2])


def model2loss(strength=1.):
	penalties = np.array([
		[[1., 1.],
		[strength, 1.]],
		[[1., strength],
		[1., 1.]],
	])

	penalties = tf.constant(penalties, dtype=tf.float32)

	def fn(y_true, ll):
		x_protected = ll[:, 2:]
		y_pred = ll[:, :2]

		base = keras.losses.categorical_crossentropy(y_true, y_pred)

		x_protected = tf.argmax(x_protected, axis=1)
		y_pred = tf.argmax(y_pred, axis=1)
		y_true = tf.argmax(y_true, axis=1)

		penalty_indices = tf.stack([x_protected, y_true, y_pred], axis=1)
		penalty = tf.gather_nd(penalties, penalty_indices)

		return base * penalty

	return fn


def getEnsembleModel(models, model_input):
	# outputs = [model.outputs[0] for model in models]
	out0 = getQualifiedLayer(models[0], model_input, len(models[0].layers) - 2)
	out1 = getQualifiedLayer(models[1], model_input, len(models[1].layers) - 2)
	out2 = getQualifiedLayer(models[2], model_input, len(models[2].layers) - 2)

	# x = models[0].layers[0](model_input)
	# for c in range(1, len(models[0].layers)):
	# 	x = model.layers[c](x)

	print(out0)
	print(out1)
	print(out2)
	output = concatenate([out0, out1, out2])
	print(model_input)
	x = Dense(256, activation='relu')(output)
	x = Dense(196, activation='relu')(x)
	x = Dense(96, activation='relu')(x)
	x = Dense(64, activation='relu')(x)
	x = Dense(10, activation='softmax')(x)

	model = Model(input=model_input, output=x, name='test')
	return model


def getAveragingEnsemble(models, modelInput):
	outputs = []
	for c in range(len(models)):
		output = getQualifiedLayer(models[c], modelInput, len(models[c].layers) - 1)
		outputs.append(output)
	y = keras.layers.Average()(outputs)
	model = Model(modelInput, y, name='ensemble')
	return model


def getLastLayerConcatenatedEnsemble(models, modelInput):
	outputs = []
	for c in range(len(models)):
		output = getQualifiedLayer(models[c], modelInput, len(models[c].layers) - 1)
		outputs.append(output)
		models[c].trainable = False
	output = concatenate(outputs)

	x = Dense(4, activation='relu')(output)
	x = Dense(2, activation='softmax')(x)

	model = Model(input=modelInput, output=x, name='test')
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
	return model


def getMiddleConcatenatedEnsemble(models, modelInput):
	outputs = []
	for c in range(len(models)):
		output = getQualifiedLayer(models[c], modelInput, len(models[c].layers) - 2)
		outputs.append(output)
	output = concatenate(outputs)

	x = Dense(32, activation='relu')(output)
	x = Dense(20, activation='relu')(x)
	x = Dense(10, activation='relu')(x)
	x = Dense(2, activation='softmax')(x)

	model = Model(input=modelInput, output=x, name='test')
	return model


def getQualifiedLayer(model, modelInput, layerIndex):
	x = modelInput
	for c in range(1, layerIndex + 1):
		x = model.layers[c](x)
	return x


def readData(path, trainTestSplit=0.5, trainTestIndex=None):
	data = pd.read_csv(path, skiprows=0)
	dataShape = data.values.shape

	if trainTestIndex is not None:
		test_idx = trainTestIndex
	else:
		test_idx = int(round(dataShape[0] * trainTestSplit))

	X_train_data = np.zeros((data.values[:test_idx].shape[0], 420))
	X_test_data = np.zeros((data.values[test_idx:].shape[0], 420))

	X_train_data[:, :409] = data.values[:test_idx, 1:-2]
	X_test_data[:, :409] = data.values[test_idx:, 1:-2]
	Y_train_data = data.values[:test_idx, -2:]
	Y_test_data = data.values[test_idx:, -2:]
	Y_train_data = np.argmax(Y_train_data, axis=1)
	Y_test_data = np.argmax(Y_test_data, axis=1)

	print(X_train_data.shape, Y_train_data.shape)
	print(X_test_data.shape, Y_test_data.shape)
	result = dict({'data': data,
	                'test_index': test_idx,
	                'x_train': X_train_data,
	                'y_train': Y_train_data,
	                'x_test': X_test_data,
	                'y_test': Y_test_data,
	               'columns': list(data.columns[1:])})
	return result


def getData(path, trainTestSplit=0.5, trainTestIndex=None):
	data = pd.read_csv(path, skiprows=0)
	dataShape = data.values.shape

	if trainTestIndex is not None:
		test_idx = trainTestIndex
	else:
		test_idx = int(round(dataShape[0] * trainTestSplit))

	modelData = data.values[:, 1:]
	model2X = modelData[:, :-2]
	model2X = MinMaxScaler().fit_transform(model2X)
	model2Y = modelData[:, -2:]
	X_train_data, Y_train_data = model2X[:test_idx], model2Y[:test_idx]
	X_test_data, Y_test_data = model2X[test_idx:], model2Y[test_idx:]
	modelInput = Input((X_train_data.shape[1],))

	print(X_train_data.shape, Y_train_data.shape)
	print(X_test_data.shape, Y_test_data.shape)
	result = dict({'data': data,
	                'test_index': test_idx,
	                'x_train': X_train_data,
	                'y_train': Y_train_data,
	                'x_test': X_test_data,
	                'y_test': Y_test_data,
	               'input': modelInput,
	               'columns': list(data.columns[1:])})
	return result


def moveValuesToMean(values, y_label):
	mean_female = np.mean(values[y_label[:, 0] == 1], axis=0)
	mean_male = np.mean(values[y_label[:, 0] == 0], axis=0)
	var_female = np.sqrt(np.var(values[y_label[:, 0] == 1], axis=0))
	var_male = np.sqrt(np.var(values[y_label[:, 0] == 0], axis=0))
	return values - np.array([mean_female if y[0] == 1 else mean_male for y in y_label])


def setTrainability(model, trainable):
	model.trainable = trainable
	for layer in model.layers:
		layer.trainable = trainable


def unique(seq):
	seen = set()
	return [x for x in seq if x not in seen and not seen.add(x)]


def main():
	loadedData = getData('C:/Users/Main/Documents/Data/Census Income/census-income.categorical.csv', trainTestIndex=199522)

	data = loadedData['data']
	test_idx = loadedData['test_index']
	x_train = loadedData['x_train']
	y_train = loadedData['y_train']
	x_test = loadedData['x_test']
	y_test = loadedData['y_test']
	columns = loadedData['columns']
	scaler = MinMaxScaler()
	scaler.fit(x_train)
	x_train = scaler.transform(x_train)
	x_test = scaler.transform(x_test)
	print(x_train.shape, x_train.shape)
	print(x_test.shape, x_test.shape)
	featureMap = [i.split('==', 1)[0] for i in columns]
	featureNames = [i.split('==', 1)[-1] for i in columns]
	features = unique(featureMap)
	featureStartIndices = []
	featureEndIndices = []
	for feature in features:
		thisFeatureStartColumnIndex = featureMap.index(feature)
		thisFeatureEndColumnIndex = len(featureMap) - 1 - featureMap[::-1].index(feature)
		protectedIndices = list(range(thisFeatureStartColumnIndex, thisFeatureEndColumnIndex + 1))
		#protectedIndices = [103, 104]
		protectedNames = [featureNames[i] for i in protectedIndices]

		if len(protectedIndices) > 1:
			modelInput = Input(x_train.shape[1:])
			modelClassWeight = {0: 1., 1: y_train.shape[0] / np.sum(np.argmax(y_train, axis = 1))}
			M2Trainer, M2Model = getM2Model(modelInput, protectedIndices = protectedIndices)
			trainModel(model = M2Trainer, num_epochs = 1, filename = 'model_m2.weights', x_train = x_train,
		           y_train = y_train, batch_size = 2048, class_weight = modelClassWeight, validation_split = 0.1)
			M2Trainer.load_weights('weights/model_m2.weights')
			print('Error for model 2: ' + repr(evaluate_error(M2Model, x_test, y_test)))
			quantifyConfusion(M2Model, x_test, y_test, protectedIndices, indexNames = protectedNames)
		#featureStartIndices.append(featureMap.index(feature))
		#featureEndIndices.append(len(featureMap) - 1 - featureMap[::-1].index(feature))

	# indicies_protected = list(columns.index(x) for x in ["sex==Female", "sex==Male"])

	# Feature Indices
	# maleColumnIndex = 105 - 1
	# femaleColumnIndex = 104 - 1
	# black = 91 - 1
	# white = 93 - 1

	# modelInput = Input(x_train.shape[1:])
	#sample_weight = compute_sample_weight("balanced", np.concatenate([np.expand_dims(x_train[:, femaleColumnIndex], axis=1)
	#			                                                                 , np.expand_dims(y_train, axis=1)], axis=1))
	# model1ClassWeight = {0: 1., 1: y_train.shape[0] / np.sum(np.argmax(y_train, axis=1))}
	# idx = 199522
	# model2Data = data.values[:, 1:]
	# model2X = model2Data[:, :-2]
	# model2X = MinMaxScaler().fit_transform(model2X)
	# model2Y = model2Data[:, -2:]
	# model2XTrain, model2YTrain = model2X[:idx], model2Y[:idx]
	# model2XTest, model2YTest = model2X[idx:], model2Y[idx:]
	# model2Input = Input((model2XTrain.shape[1],))
	# classes = range(y_train.shape[1])
	# model2ClassWeight = sklearn.utils.class_weight.compute_class_weight("balanced", classes, np.argmax(y_train, axis=1))
	# model2ClassWeight = dict(zip(classes, model2ClassWeight))
	#
	# x_train_tmp = x_train.copy()
	# x_test_tmp = x_test.copy()
	# x_train_tmp[:, femaleColumnIndex:femaleColumnIndex+2] = 0
	# y_label = x_train[:, femaleColumnIndex:femaleColumnIndex+2]
	# y_label_te = x_test[:, femaleColumnIndex:femaleColumnIndex+2]

	#baseModel = getBaseModel(inp)
	#autoencoderModel = getAutoencoderModel(inp)
	# M1Trainer, M1Encoder = getM1Model(modelInput)
	# M2Trainer, M2Model = getM2Model(modelInput, protectedIndices = indicies_protected)
	#
	# retrainModel1 = False
	#Train models
	# if(retrainModel1):
		#trainModel(model=baseModel, num_epochs=100, filename='model_base.weights', x_train=x_train, y_train=to_categorical(y_train), batch_size=2048, class_weight=class_weight)
		#trainModel(model=baseModel, num_epochs=100, filename='model_autoencoder.weights', x_train=x_train, y_train=to_categorical(y_train), batch_size=2048, class_weight=class_weight)
	# 	trainModel(model=M1Trainer, num_epochs=20, filename='model_m1.weights', x_train=[x_train_tmp, y_label], y_train=[x_train_tmp, y_label], batch_size=2048, class_weight=model1ClassWeight, validation_data = [[x_test_tmp,y_label_te], [x_test_tmp, y_label_te]])
	#
	# retrainModel2 = False
	# if(retrainModel2):
	# 	trainModel(model = M2Trainer, num_epochs = 50, filename = 'model_m2.weights', x_train = x_train,
	# 	           y_train = y_train, batch_size = 2048, class_weight = model1ClassWeight, validation_split = 0.1)

	#Load models
	#baseModel.load_weights('weights/model_base.weights')
	#autoencoderModel.load_weights('weights/model_autoencoder.weights')
	# M1Trainer.load_weights('weights/model_m1.weights')
	#M1Trainer.summary()
	# M2Trainer.load_weights('weights/model_m2.weights')
	#M2Trainer.summary()

	#Fix Encoder Module
	# encoderTrainValues = M1Encoder.predict(x_train)
	# encoderTestValues = M1Encoder.predict(x_test)
	# encoderTrainValues = moveValuesToMean(encoderTrainValues, y_label)
	# encoderTestValues = moveValuesToMean(encoderTestValues, y_label_te)
	# n_dim = 64
	# inp = Input((n_dim,))
	# x = Dropout(0.5)(inp)
	# x = Dense(32, activation='relu')(x)
	# out = Dense(2, activation='softmax')(x)
	#
	# M1Classifier = Model(inp, out, name='m1classifier')
	# M1Classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
	#trainModel(model=M1Classifier, num_epochs=100, filename='model_autoencoder.weights', x_train=encoderTrainValues, y_train=to_categorical(np.argmax(y_train, axis=1)), batch_size=2048, class_weight=model1ClassWeight, validation_split=0.1)
	#M1Classifier.summary()
	# M1Classifier.load_weights('weights/model_autoencoder.weights')
	# setTrainability(M1Classifier, False)
	# M1Model = stackModels(modelInput, M1Encoder, M1Classifier)
	# M1Model.trainable = False
	# M2Model.trainable = False
	# setTrainability(M2Model, False)
	# M1Model.summary()
	# M2Model.summary()
	#M1Model.summary()
	#quantifyConfusion(baseModel, x_test, y_test)
	#quantifyConfusion(autoencoderModel, x_test, y_test)
	#quantifyConfusion(M1Model, [x_test_tmp, y_label], [x_test_tmp, y_label])

	#model = keras.models.load_model('C:/Users/Main/Documents/PyCharm/Machine-Learning/weights/acc69.25_dis0.185.h5')
	#encoder = keras.models.load_model('C:/Users/Main/Documents/PyCharm/Machine-Learning/weights/acc69.25_dis0.185_encoder.h5')
	#model.summary()
	#encoder.summary()

	#---------------------------------------
	# models = [M1Model, M2Model]
	# averagingEnsemble = getAveragingEnsemble(models, modelInput)
	#
	# print('Error for model 1: ' + repr(evaluate_error(M1Model, x_test, y_test)))
	# quantifyConfusion(M1Model, x_test, y_test, femaleColumnIndex, maleColumnIndex)
	#
	# outputConcatEnsemble = getLastLayerConcatenatedEnsemble(models, modelInput)
	# trainOutputConcatEnsemble = False
	# if trainOutputConcatEnsemble:
	# 	outputConcatEnsemble.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
	# 	trainModel(model=outputConcatEnsemble, x_train=x_train, y_train=y_train, num_epochs=20, batch_size=2048, class_weight=model1ClassWeight, validation_split=0.1, filename='model_output_concat_ensemble.weights')
	# outputConcatEnsemble.load_weights('weights/model_output_concat_ensemble.weights')
	#
	# middleConcatEnsemble = getMiddleConcatenatedEnsemble(models, modelInput)
	# trainMiddleConcatEnsemble = True
	# if trainMiddleConcatEnsemble:
	# 	middleConcatEnsemble.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
	# 	trainModel(model=middleConcatEnsemble, x_train=x_train, y_train=y_train, num_epochs=20, batch_size=2048,
	# 	           class_weight=model1ClassWeight, validation_split=0.1,
	# 	           filename='model_middle_concat_ensemble.weights')
	# middleConcatEnsemble.load_weights('weights/model_middle_concat_ensemble.weights')


	# print('Error for model 1: ' + repr(evaluate_error(M1Model, x_test, y_test)))
	# quantifyConfusion(M1Model, x_test, y_test, femaleColumnIndex, maleColumnIndex)
	# print('Error for model 2: ' + repr(evaluate_error(M2Model, x_test, y_test)))
	# quantifyConfusion(M2Model, x_test, y_test, indicies_protected)
	# print('Error for simple ensemble: ' + repr(evaluate_error(averagingEnsemble, x_test, y_test)))
	# quantifyConfusion(averagingEnsemble, x_test, y_test, femaleColumnIndex, maleColumnIndex)
	# print('Error for last layer ensemble: ' + repr(evaluate_error(outputConcatEnsemble, x_test, y_test)))
	# quantifyConfusion(outputConcatEnsemble, x_test, y_test, femaleColumnIndex, maleColumnIndex)
	# print('Error for middle layer ensemble: ' + repr(evaluate_error(middleConcatEnsemble, x_test, y_test)))
	# quantifyConfusion(middleConcatEnsemble, x_test, y_test, femaleColumnIndex, maleColumnIndex)
	# print('end')
	#---------------------------------------

main()

# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_train = x_train / 255.
# x_test = x_test / 255.
# y_train = to_categorical(y_train, num_classes=10)
#
# print(
# 	'x_train shape: {} | y_train shape: {}\nx_test shape : {} | y_test shape : {}'.format(x_train.shape, y_train.shape,
# 	                                                                                      x_test.shape, y_test.shape))
#
# input_shape = x_train[0, :, :, :].shape
# model_input = Input(shape=input_shape)
#
# model_0 = conv_pool_cnn(model_input)
# model_1 = all_cnn(model_input)
# model_2 = nin_cnn(model_input)
#
# #_ = compile_and_train(model_0, num_epochs=20, filename='model_0.hdf5')
# #_ = compile_and_train(model_1, num_epochs=20, filename='model_1.hdf5')
# #_ = compile_and_train(model_2, num_epochs=20, filename='model_2.hdf5')
#
# model_0.load_weights('weights/model_0.hdf5')
# model_1.load_weights('weights/model_1.hdf5')
# model_2.load_weights('weights/model_2.hdf5')
#
# models = [model_0, model_1, model_2]


# ensemble = getEnsembleModel(models, model_input)
# simpleEnsemble = simpleAveragingEnsemble(models, model_input)
# compile_and_train(ensemble, 10, 'ensemble.hdf5')
# print('Error for model 0: ' + repr(evaluate_error(model_0)))
# print('Error for model 1: ' + repr(evaluate_error(model_1)))
# print('Error for model 2: ' + repr(evaluate_error(model_2)))
# print('Error for simple ensemble: ' + repr(evaluate_error(simpleEnsemble)))
# print('Error for complex ensemble: ' + repr(evaluate_error(ensemble)))
