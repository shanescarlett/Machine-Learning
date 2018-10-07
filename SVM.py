import os

import sklearn
from sklearn import cross_validation, grid_search
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.externals import joblib


def train_svm_classifer(features, labels, model_output_path):
	"""
	train_svm_classifer will train a SVM, saved the trained and SVM model and
	report the classification performance

	features: array of input features
	labels: array of labels associated with the input features
	model_output_path: path for storing the trained svm model
	"""
	# save 20% of data for performance evaluation
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(features, labels, test_size = 0.2)

	param = [
		{
			"kernel": ["linear"],
			"C": [1, 10, 100, 1000]
		},
		{
			"kernel": ["rbf"],
			"C": [1, 10, 100, 1000],
			"gamma": [1e-2, 1e-3, 1e-4, 1e-5]
		}
	]

	# request probability estimation
	svm = SVC(probability = True)

	# 10-fold cross validation, use 4 thread as each fold and each parameter set can be train in parallel
	clf = grid_search.GridSearchCV(svm, param,
	                               cv = 10, n_jobs = 4, verbose = 3)

	clf.fit(X_train, y_train)

	if os.path.exists(model_output_path):
		joblib.dump(clf.best_estimator_, model_output_path)
	else:
		print("Cannot save trained svm model to {0}.".format(model_output_path))

	print("\nBest parameters set:")
	print(clf.best_params_)

	y_predict = clf.predict(X_test)

	labels = sorted(list(set(labels)))
	print("\nConfusion matrix:")
	print("Labels: {0}\n".format(",".join(labels)))
	print(confusion_matrix(y_test, y_predict, labels = labels))

	print("\nClassification report:")
	print(classification_report(y_test, y_predict))


def readData(path, trainTestSplit=0.5, trainTestIndex=None):
	import pandas as pd
	import numpy as np
	data = pd.read_csv(path, skiprows=0)
	dataShape = data.values.shape

	if trainTestIndex is not None:
		test_idx = trainTestIndex
	else:
		test_idx = int(round(dataShape[0] * trainTestSplit))

	X_train_data = np.zeros((data.values[:test_idx].shape[0], 420))
	X_test_data = np.zeros((data.values[test_idx:].shape[0], 420))

	X_train_data = data.values[:test_idx, :]
	X_test_data = data.values[test_idx:, :]
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


def main():
	print('hi')
	data = readData('C:/Users/Main/Documents/Data/Bearing/1st_test/1st_test/2003.10.22.12.06.24', trainTestSplit = 0.66)


main()
print('End')

