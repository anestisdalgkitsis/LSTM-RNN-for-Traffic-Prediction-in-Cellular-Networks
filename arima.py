#
# arima.py
# AutorRegressive Integrated Moving Average Engine | Version 1.13.1
#
# Created by Anestis Dalgkitsis in 25/08/2017.
# Copyright 2017 Anestis Dalgkitsis. All rights reserved.
#
# Contact: anestisdalgkitsis@gmail.com
#

# Depentencies

import time

from statsmodels.tsa.arima_model import ARIMA

from sklearn.metrics import mean_squared_error

# Functions

# One Step Forecast
def forecast(observations, x, lag = 5, difference = 1, model = 0):
	model = ARIMA(observations, order = (lag, difference, model))
	modelfit = model.fit(disp=0) # Update model
	output = modelfit.forecast() # Predict updated model
	return output[0]

# Forecast Functions

def arima(dates, data, measurement, lag = 5, difference = 1, model = 0):

    starttime = time.time()

    X = data
    train, test = X[0:len(X)], X[0:len(X)]
    observations = [x for x in train] # We keep track of all observations in this list that is seeded with the training data and to which new observations are appended each iteration.
    predictions = list()

    print(" [i] Forecasting")
    for t in range(len(test)):
    	predictions.append(forecast(observations, t, lag, difference, model))
    	obs = test[t]
    	observations.append(obs)
    	print('  |- Day = %d/%d, Predicted = %f, Expected = %f' % (t + 1, len(test), predictions[t], obs))

    print(" [i] Report")
    duration = time.time() - starttime
    print('  |- Duration (s) : ', duration)
    error = mean_squared_error(measurement, predictions)
    print('  |- MSE: %.3f' % error)

    return predictions, error, duration

def nextday(dates, data, x, lag = 5, difference = 1, model = 0):

    X = data
    train, test = X[0:len(X)], X[0:len(X)+1]
    observations = [x for x in train] # We keep track of all observations in this list that is seeded with the training data and to which new observations are appended each iteration.

    for t in range(len(test)):
    	obs = test[t]
    	observations.append(obs)

    return forecast(observations, len(test)+1, lag, difference, model)[0]

def forecastarima(dates, data, measurement, split = 0.6, lag = 5, difference = 1, model = 0):

	# Split dataset
	trainsize = int(len(data) * split)
	testsize = len(data) - trainsize
	train, test = data[0:trainsize:], data[trainsize:len(data):]

	starttime = time.time()

	observations = [x for x in train] # We keep track of all observations in this list that is seeded with the training data and to which new observations are appended each iteration.
	predictions = list()

	print(" [i] Forecasting")
	for i in range(len(train)):
		print('  |- Day = %d/%d, Predicted = X, Expected = %f' % (i + 1, len(data), data[i]))
	for t in range(len(test)):
		predictions.append(forecast(observations, t, lag, difference, model))
		observations.append(predictions[t][0])
		print('  |- Day = %d/%d, Predicted = %f, Expected = %f' % (len(train) + t + 1, len(data), predictions[t], test[t]))

	print(" [i] Report")
	duration = time.time() - starttime
	print('  |- Duration (s) : ', duration)
	error = mean_squared_error(measurement[trainsize:len(data):], predictions)
	print('  |- MSE: %.3f' % error)

	return predictions, error, duration
