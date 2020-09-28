#
# rbf.py
# Radial Basis Function Compute Engine | Version 3.20.3
#
# Created by Anestis Dalgkitsis in 23/02/2017.
# Copyright 2017 Anestis Dalgkitsis. All rights reserved.
#
# Contact: anestisdalgkitsis@gmail.com | +30 6981975737
#

# Depentencies

import time

import numpy
from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error

# Functions

# One Step Forecast
def forecast(dates, data, x, C = 1e3, gamma = 0.1):

    dates = numpy.reshape(dates, (len(dates), 1))

    # Init SVR
    svrrbf = SVR(kernel = 'rbf', C = C, gamma = gamma)

    # Train Model
    svrrbf.fit(dates, data)

    return svrrbf.predict(x)[0]

# Forecast Functions

# Evaluation with RBF
def rbf(dates, data, measurement, C = 1e3, gamma = 0.1):

    # Step-by-step Forecasting

    print(" [i] Predicting...")

    starttime = time.time()

    predictions = list()
    for i in range(0, len(dates)):
        predictions.append(forecast(dates, data, i, C, gamma))
        print('  |- Day = %d/%d, Predicted = %f, Expected = %f' % (i + 1, len(dates), predictions[i], data[i]))

    # Report Performance

    print(" [i] Report")
    duration = time.time() - starttime
    print('  |- Duration (s) : ', duration)
    error = mean_squared_error(measurement, predictions)
    print('  |- MSE: %.3f' % error)

    return predictions, error, duration

def forecastrbf(dates, data, measurement, split, C = 1e3, gamma = 0.1):

    # Split dataset
    trainsize = int(len(data) * split)
    testsize = len(data) - trainsize
    train, test = data[0:trainsize:], data[trainsize:len(data):]

    # Step-by-step Forecasting

    print(" [i] Predicting...")

    for i in range(len(train)):
            print('  |- Day = %d/%d, Predicted = X, Expected = %f' % (i + 1, len(data), data[i]))

    starttime = time.time()

    predictions = list()
    for i in range(0, len(test)):
        predictions.append(forecast(dates[0:trainsize:], train, i, C, gamma))
        print('  |- Day = %d/%d, Predicted = %f, Expected = %f' % (len(train) + i + 1, len(dates), predictions[i], data[i]))

    # Report Performance

    print(" [i] Report")
    duration = time.time() - starttime
    print('  |- Duration (s) : ', duration)
    error = mean_squared_error(measurement[trainsize:len(data):], predictions)
    print('  |- MSE: %.3f' % error)

    return predictions, error, duration
