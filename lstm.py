#
# lstm.py
# Long-Short Term Memory Recurrent Neural Network Engine | Version 5.2.4
#
# Created by Anestis Dalgkitsis in 03/07/2017.
# Copyright 2017 Anestis Dalgkitsis. All rights reserved.
#
# Contact: anestisdalgkitsis@gmail.com
#

# Depentencies

import os
import time
import warnings

import numpy
import math
from pandas import DataFrame
from pandas import Series
from pandas import concat
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Silence Warnings

warnings.filterwarnings("ignore") # Silence Numpy Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Silence TensorFlow Warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning) # Silence Pandas Future Warnings

# Utility Functions

# Transform time series into a supervised learning problem
def transformtimeseriestosupervised(data, lag = 1):
	dataframe = DataFrame(data)
	columns = [dataframe.shift(i) for i in range(1, lag + 1)]
	columns.append(dataframe)
	dataframe = concat(columns, axis = 1)
	dataframe.fillna(0, inplace = True)
	return dataframe

# Transform the time series data so that it is stationary by differencing the data.
def difference(dataset, interval = 1):
	differenced = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		differenced.append(value)
	return Series(differenced)

# Invert differenced values in order to take forecasts back into their original scale.
def inversedifference(history, yhat, interval = 1):
	return yhat + history[-interval]

# Transform observations data to [-1, 1] in irder to be within the scale of the activation function (tanh) used by the LSTM network.
def scale(train, test):
	# fit scaler
	scaler = MinMaxScaler(feature_range = (-1, 1))
	scaler = scaler.fit(train)
	# transform train
	train = train.reshape(train.shape[0], train.shape[1])
	trainscaled = scaler.transform(train)
	# transform test
	test = test.reshape(test.shape[0], test.shape[1])
	testscaled = scaler.transform(test)
	return scaler, trainscaled, testscaled

# Inverse scaling in order to take forecasted values back into their original scale
def invertscale(scaler, X, value):
	newrow = [x for x in X] + [value]
	array = numpy.array(newrow)
	array = array.reshape(1, len(array))
	inverted = scaler.inverse_transform(array)
	return inverted[0, -1]

# Fit LSTM network to the training data
def fit(train, batchsize, epochs, neurons):
	# Reshape data into (Samples, Time Steps, Features) for the LSTM Layer matrix
	X, y = train[:, 0:-1], train[:, -1]
	X = X.reshape(X.shape[0], 1, X.shape[1])
	# Build LSTM Model using the Sequential Keras API for TensorFlow Backend
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape = (batchsize, X.shape[1], X.shape[2]), stateful = True))
	model.add(Dense(1))
	model.compile(loss = 'mean_squared_error', optimizer = 'adagrad') # mse | rmsprop OLD
	# Fit the network to the training data
	for epoch in range(epochs):
		model.fit(X, y, epochs = 1, batch_size = batchsize, verbose = 0, shuffle = False)
		model.reset_states()
		print('  |- Epoch = %d/%d' % (epoch + 1, epochs))
	return model

# One Step Forecast
def forecast(model, batch_size, X):
	X = X.reshape(1, 1, len(X))
	yhat = model.predict(X, batch_size = batch_size)
	return yhat[0,0]

# Forecast Functions

def evaluationlstm(data, measurement, batch = 1, epochs = 1500, neurons = 2):

	starttime = time.time()

	print(" [i] Normalizing Dataset")

	# Transform data to be stationary
	differencedvalues = difference(data, 1)

	# transform data to be supervised learning
	supervised = transformtimeseriestosupervised(differencedvalues, 1)
	supervisedvalues = supervised.values

	# Use the whole dataset for train and test-sets
	train, test = supervisedvalues[0:len(supervisedvalues)], supervisedvalues[0:len(supervisedvalues)]

	# Transform the scale of the data
	scaler, trainscaled, testscaled = scale(train, test)

	print(" [i] Training Phase")
	fitstarttime = time.time()

	# Fit the model
	model = fit(trainscaled, batch, epochs, neurons)

	fitduration = time.time() - fitstarttime

	# Walk-Forward Model Validation

	print(" [i] Testing Phase")

	predictionstarttime = time.time()

	predictions = list()
	for i in range(len(testscaled)):
		# One-step forecast
		X, y = testscaled[i, 0:-1], testscaled[i, -1]
		yhat = forecast(model, 1, X)
		# Invert scaling
		yhat = invertscale(scaler, X, yhat)
		# Invert differencing
		yhat = inversedifference(data, yhat, len(testscaled) + 1 - i)
		# Store forecast
		predictions.append(yhat)
		print('  |- Day = %d/%d, Predicted = %f, Expected = %f' % (i + 1, len(data[0:len(data) - 1]), yhat, data[i]))

	predictionduration = time.time() - predictionstarttime

	# Report Performance

	print(" [i] Report")
	print('  |- Training Phase Duration (s) : ', fitduration)
	print('  |- Testing Phase Duration (s)  : ', predictionduration)
	totalduration = time.time() - starttime
	print('  |- Total Duration (s)          : ', totalduration)
	mse = mean_squared_error(measurement[0:len(data) - 1], predictions)
	print('  |- MSE                         :  %.3f' % mse)
	return predictions, mse, totalduration, fitduration, predictionduration

def nextday(dates, data, x, batch = 1, epochs = 1500, neurons = 2):

	print(" [i] Normalizing Data")

	# Transform data to be stationary
	differencedvalues = difference(data, 1)

	# transform data to be supervised learning
	supervised = transformtimeseriestosupervised(differencedvalues, 1)
	supervisedvalues = supervised.values

	# Use the whole dataset for train and test-sets
	train, test = supervisedvalues[0:len(supervisedvalues)], supervisedvalues[0:len(supervisedvalues)]

	# Transform the scale of the data
	scaler, trainscaled, testscaled = scale(train, test)

	print(" [i] Fitting Model")

	# Fit the model
	model = fit(trainscaled, batch, epochs, neurons)

	print(" [i] Predicting Model")

	# One-step forecast for the next of the last dataset day
	X = testscaled[len(testscaled) - 1, -1:]
	yhat = forecast(model, 1, X)
	# Invert scaling
	yhat = invertscale(scaler, X, yhat)
	# Invert differencing
	yhat = inversedifference(data, yhat, 1)

	return yhat

def forecastlstm(data, measurement, split, batch = 1, epochs = 1500, neurons = 2):

	# Split dataset
	trainsize = int(len(data) * split)
	testsize = len(data) - trainsize

	starttime = time.time()

	print(" [i] Normalizing Dataset")

	# Transform data to be stationary
	differencedvalues = difference(data, 1)

	# transform data to be supervised learning
	supervised = transformtimeseriestosupervised(differencedvalues, 1)
	supervisedvalues = supervised.values

	# Use the splited dataset for train and then for test
	train, test = supervisedvalues[0:trainsize:], supervisedvalues[trainsize:len(supervisedvalues):]

	# Transform the scale of the data
	scaler, trainscaled, testscaled = scale(train, test)

	print(" [i] Training Phase")
	fitstarttime = time.time()

	# Fit the model
	model = fit(trainscaled, batch, epochs, neurons)

	fitduration = time.time() - fitstarttime

	# Walk-Forward Model Validation

	print(" [i] Testing Phase")

	for i in range(len(train)):
		print('  |- Day = %d/%d, Predicted = X, Expected = %f' % (i + 1, len(data[0:len(data) - 1]), data[i]))

	predictionstarttime = time.time()

	predictions = list()
	for i in range(len(testscaled)):
		# One-step forecast
		X, y = testscaled[i, 0:-1], testscaled[i, -1]
		yhat = forecast(model, 1, X)
		# Invert scaling
		yhat = invertscale(scaler, X, yhat)
		# Invert differencing
		yhat = inversedifference(data[0:trainsize:], yhat, len(testscaled) + 1 - i)
		# Store forecast
		predictions.append(yhat)
		print('  |- Day = %d/%d, Predicted = %f, Expected = %f' % (len(train) + i + 1, len(data[0:len(data) - 1]), yhat, data[i]))

	predictionduration = time.time() - predictionstarttime

	# Report Performance

	print(" [i] Report")
	print('  |- Training Phase Duration (s) : ', fitduration)
	print('  |- Testing Phase Duration (s)  : ', predictionduration)
	totalduration = time.time() - starttime
	print('  |- Total Duration (s)          : ', totalduration)
	error = mean_squared_error(measurement[trainsize:len(data) - 1:], predictions)
	print('  |- MSE                         :  %.3f' % error)
	return predictions, error, totalduration, fitduration, predictionduration

# sLSTM Forecasting Function

def slstm(data, measurement, split, blocksNumber = 4, batch = 1, epochs = 1500, neurons = 2):

	# Split dataset
	trainsize = int(len(data) * split)
	testsize = len(data) - trainsize

	starttime = time.time()

	print(" [i] Searching Optimal Dataset Block")

	trainDataset = data[0:trainsize:]
	blockSize = math.floor(len(trainDataset) / blocksNumber)
	winner = -1
	minRSD = 99999.99
	for i in range(0, blocksNumber):
		# Standard Deviation
		blockSum = sum(trainDataset[(i * blockSize):((i + 1) * blockSize)])
		BlockMean = blockSum / blockSize
		blockSquaredSum = 0
		for j in range((i * blockSize), len(trainDataset[0:((i + 1) * blockSize)])):
			blockSquaredSum = pow((trainDataset[j] - BlockMean), 2) + blockSquaredSum
		blockSquaredMean = blockSquaredSum / blockSize
		StandardDeviation = math.sqrt(blockSquaredMean)
		# Relative Standard Deviation
		blockRSD = (StandardDeviation * 100) / BlockMean
		print('  |- #%d Block RSD : %3f' % (i, blockRSD))
		if minRSD >= blockRSD:
			winner = i
			minRSD = blockRSD

	print(' [i] Using Block #%d as Training Dataset' % winner)
	winnerDataset = trainDataset[(winner * blockSize):((winner + 1) * blockSize)]

	print(" [i] Normalizing Dataset")

	# Transform data to be stationary
	differencedvalues = difference(data, 1)

	# transform data to be supervised learning
	supervised = transformtimeseriestosupervised(differencedvalues, 1)
	supervisedvalues = supervised.values

	# Transform block data to be stationary
	differencedvalues = difference(winnerDataset, 1)

	# transform block data to be supervised learning
	winnerSupervised = transformtimeseriestosupervised(differencedvalues, 1)
	winnerSupervisedValues = winnerSupervised.values

	# Use the splited dataset for train and then for test
	train, test = winnerSupervisedValues[0:len(winnerDataset):], supervisedvalues[trainsize:len(supervisedvalues):]

	# Transform the scale of the data
	scaler, trainscaled, testscaled = scale(train, test)

	print(" [i] Training Phase")
	fitstarttime = time.time()

	# Fit the model
	model = fit(trainscaled, batch, epochs, neurons)

	fitduration = time.time() - fitstarttime

	# Walk-Forward Model Validation

	print(" [i] Testing Phase")

	for i in range(len(train)):
		print('  |- Day = %d/%d, Predicted = X, Expected = %f' % (i + 1, len(data[0:len(data) - 1]), data[i]))

	predictionstarttime = time.time()

	predictions = list()
	for i in range(len(testscaled)):
		# One-step forecast
		X, y = testscaled[i, 0:-1], testscaled[i, -1]
		yhat = forecast(model, 1, X)
		# Invert scaling
		yhat = invertscale(scaler, X, yhat)
		# Invert differencing
		yhat = inversedifference(data[0:trainsize:], yhat, len(testscaled) + 1 - i)
		# Store forecast
		predictions.append(yhat)
		print('  |- Day = %d/%d, Predicted = %f, Expected = %f' % (len(train) + i + 1, len(data[0:len(data) - 1]), yhat, data[i]))

	predictionduration = time.time() - predictionstarttime

	# Report Performance

	print(" [i] Smart LSTM Report Card")
	print('  |- Training Phase Duration (s) : ', fitduration)
	print('  |- Testing Phase Duration (s)  : ', predictionduration)
	totalduration = time.time() - starttime
	print('  |- Total Duration (s)          : ', totalduration)
	print('  |- Optimal Dataset Block (#)   : ', winner)
	error = mean_squared_error(measurement[trainsize:len(data) - 1:], predictions)
	print('  |- MSE                         :  %.3f' % error)
	return predictions, error, totalduration, fitduration, predictionduration
