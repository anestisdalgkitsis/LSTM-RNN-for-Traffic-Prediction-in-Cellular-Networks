#
# sarimax.py
# Seasonal AutoRegressive Integrated Moving Averages with eXogenous regressors Engine | Version 2.1.1
#
# Created by Anestis Dalgkitsis in 01/09/2017.
# Copyright 2017 Anestis Dalgkitsis. All rights reserved.
#
# Contact: anestisdalgkitsis@gmail.com
#

# Depentencies

import os
import time
import warnings

import warnings
import itertools
import numpy as np
import statsmodels.api as sm

from sklearn.metrics import mean_squared_error

# Functions

# GRID SEARCH for optimal parameters --> ARIMA(p,d,q)(P,D,Q)s
def gridsearch():

    # Fill with all possible combinations
    p = d = q = range(0, 2)

    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))

    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    return seasonal_pdq, pdq

# Forecast Functions

def sarimax(dates, data, measurement):

    starttime = time.time()

    print(" [i] Grid Search optimal parameters")

    seasonal_pdq, pdq = gridsearch()

    # FIT SARIMAX

    print(" [i] Predicting...")

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(data, order = param, seasonal_order = param_seasonal, enforce_stationarity = False, enforce_invertibility = False)
                results = mod.fit(disp = False)
            except:
                continue

    # GET prediction

    pred = results.get_prediction()

    y_forecasted = pred.predicted_mean

    print(" [i] Report")
    duration = time.time() - starttime
    print('  |- Duration (s) : ', duration)
    error = mean_squared_error(measurement[1:len(dates)], y_forecasted[1:len(dates)])
    print('  |- MSE: %.3f' % error)

    return y_forecasted[1:len(dates)], error, duration

def nextday(dates, data, x):

    seasonal_pdq, pdq = gridsearch()

    # FIT SARIMAX

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(data, order = param, seasonal_order = param_seasonal, enforce_stationarity = False, enforce_invertibility = False)
                results = mod.fit(disp = False)
            except:
                continue

    # GET prediction

    pred = results.get_prediction()

    y_forecasted = pred.predicted_mean

    return y_forecasted[x - 1]

def forecastsarimax(dates, data, measurement, split = 0.6):

    # Split dataset
    trainsize = int(len(data) * split)
    testsize = len(data) - trainsize
    train, test = data[0:trainsize:], data[trainsize:len(data):]

    starttime = time.time()

    print(" [i] Grid Search Optimal Parameters")

    seasonal_pdq, pdq = gridsearch()

    # FIT SARIMAX

    print(" [i] Forecasting")

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(train[0:testsize:], order = param, seasonal_order = param_seasonal, enforce_stationarity = False, enforce_invertibility = False)
                results = mod.fit(disp = False)
            except:
                continue

    # GET prediction

    pred = results.get_prediction()

    y_forecasted = pred.predicted_mean

    print(" [i] Report")
    duration = time.time() - starttime
    print('  |- Duration (s) : ', duration)
    error = mean_squared_error(measurement[trainsize:len(data) - 1:], y_forecasted[1:len(dates)])
    print('  |- MSE: %.3f' % error)

    return y_forecasted[1:len(dates)], error, duration
