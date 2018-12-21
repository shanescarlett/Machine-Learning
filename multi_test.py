import keras
import numpy as np


def loadData():
	(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
	x_train = np.divide(x_train, 255)
	x_test = np.divide(x_test, 255)
	y_train = keras.utils.to_categorical(y_train, 10)
	y_test = keras.utils.to_categorical(y_test, 10)
	return x_train, y_train, x_test, y_test


def getModel(inputShape):
	x = keras.layers.Input(inputShape)
	inputLayer = x
	x = keras.layers.Conv2D(32, (3, 3), padding = 'same', activation = 'relu')(x)
	x = keras.layers.MaxPooling2D(pool_size = (2, 2))(x)
	x = keras.layers.Dropout(0.25)(x)
	x = keras.layers.Conv2D(256, (3, 3), padding = 'same', activation = 'relu')(x)
	x = keras.layers.MaxPooling2D(pool_size = (2, 2))(x)
	x = keras.layers.Dropout(0.25)(x)
	x = keras.layers.Conv2D(1024, (3, 3), padding = 'same', activation = 'relu')(x)
	x = keras.layers.MaxPooling2D(pool_size = (2, 2))(x)
	x = keras.layers.Dropout(0.25)(x)
	x = keras.layers.Conv2D(4096, (3, 3), padding = 'same', activation = 'relu')(x)
	x = keras.layers.MaxPooling2D(pool_size = (2, 2))(x)
	x = keras.layers.Dropout(0.25)(x)
	x = keras.layers.Flatten()(x)
	x = keras.layers.Dense(512, activation = 'relu')(x)
	x = keras.layers.Dropout(0.5)(x)
	x = keras.layers.Dense(128, activation = 'relu')(x)
	x = keras.layers.Dropout(0.5)(x)
	x = keras.layers.Dense(32, activation = 'relu')(x)
	x = keras.layers.Dropout(0.5)(x)
	x = keras.layers.Dense(10, activation = 'softmax')(x)
	outputLayer = x
	return keras.models.Model(inputLayer, outputLayer)


def getMultiGPUModel(inputShape):
	model = getModel(inputShape)
	return keras.utils.multi_gpu_model(model, gpus = 6)


def compileModel(model:keras.models.Model):
	model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['binary_accuracy', 'accuracy'])


def trainModel(model, xTrain, yTrain, xTest, yTest):
	model.summary()
	checkpoint = keras.callbacks.ModelCheckpoint('model.h5', monitor = 'val_acc', verbose = 1, save_best_only = True,
	                                             mode = 'max')
	model.fit(x = xTrain, y = yTrain, validation_data = (xTest, yTest),
	       batch_size = 128*6, epochs = 1000, verbose = 1, callbacks = [checkpoint])


def main():
	xTrain, yTrain, xTest, yTest = loadData()
	model = getMultiGPUModel(xTrain[0].shape)
	# model = getModel(xTrain[0].shape)
	compileModel(model)
	trainModel(model, xTrain, yTrain, xTest, yTest)
	print('End')


if __name__ == "__main__":
	main()