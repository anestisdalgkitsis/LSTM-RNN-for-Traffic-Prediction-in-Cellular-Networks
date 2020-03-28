#
# plot.py
# Extented Data Ploting Functions | Version 3.2.1
#
# Created by Anestis Dalgkitsis in 02/10/2017.
# Copyright 2017 Anestis Dalgkitsis. All rights reserved.
#
# Contact: anestisdalgkitsis@gmail.com
#

# Depentencies

import csv
import datetime as datetime
from dateutil.parser import parse

from matplotlib import dates as mdates
import matplotlib.pyplot as pyplot

# Utility Functions

def is_date(string):
    try:
        parse(string)
        return True
    except ValueError:
        return False

def splitcalculator(dates, data, split):
    trainsize = int(len(data) * split)
    testsize = len(data) - trainsize
    testdates = dates[trainsize:len(data):]
    test = data[trainsize:len(dates):]
    return test, testdates

# Separate Graphs

def plot(dates, datesLabel, data, dataColor, dataLabel, predictions, predictionsColor, predictionsLabel, measurement):
    pyplot.figure(predictionsLabel)
    if is_date(dates[0]):
        pyplot.plot([datetime.datetime.strptime(d,'%d/%m/%Y').date() for d in dates], data, color = dataColor, label = dataLabel)
        pyplot.plot([datetime.datetime.strptime(d,'%d/%m/%Y').date() for d in dates], predictions, color = predictionsColor, label = predictionsLabel)
    else:
        pyplot.plot(dates, data, color = dataColor, label = dataLabel)
        pyplot.plot(dates, predictions, color = predictionsColor, label = predictionsLabel)
    pyplot.xlabel(datesLabel)
    pyplot.ylabel(measurement)
    pyplot.legend()
    pyplot.show()
    return

def splitplot(dates, datesLabel, data, dataColor, dataLabel, predictions, predictionsColor, predictionsLabel, split, measurement):
    pyplot.figure(predictionsLabel)
    test, testdates = splitcalculator(dates, data, split)
    if is_date(dates[0]):
        pyplot.plot([datetime.datetime.strptime(d,'%d/%m/%Y').date() for d in testdates], test, color = dataColor, label = dataLabel, linewidth = 2.0)
        pyplot.plot([datetime.datetime.strptime(d,'%d/%m/%Y').date() for d in testdates], predictions, color = predictionsColor, label = predictionsLabel)
    else:
        pyplot.plot(testdates, test, color = dataColor, label = dataLabel)
        pyplot.plot(testdates, predictions, color = predictionsColor, label = predictionsLabel)
    pyplot.xlabel(datesLabel)
    pyplot.ylabel(measurement)
    pyplot.legend()
    pyplot.show()
    return

# Combined Graphs

def plotall(dates, datesLabel, data, dataColor, dataLabel, predictions1, predictions1Color, predictions1Label, predictions2, predictions2Color, predictions2Label, predictions3, predictions3Color, predictions3Label, predictions4, predictions4Color, predictions4Label, measurement):
    pyplot.figure('Evaluation')
    pyplot.plot([datetime.datetime.strptime(d,'%d/%m/%Y').date() for d in dates], data, color = dataColor, label = dataLabel)
    pyplot.plot([datetime.datetime.strptime(d,'%d/%m/%Y').date() for d in dates], predictions1, color = predictions1Color, label = predictions1Label)
    pyplot.plot([datetime.datetime.strptime(d,'%d/%m/%Y').date() for d in dates], predictions2, color = predictions2Color, label = predictions2Label)
    pyplot.plot([datetime.datetime.strptime(d,'%d/%m/%Y').date() for d in dates], predictions3, color = predictions3Color, label = predictions3Label)
    pyplot.plot([datetime.datetime.strptime(d,'%d/%m/%Y').date() for d in dates], predictions4, color = predictions4Color, label = predictions4Label)
    pyplot.xlabel(datesLabel)
    pyplot.ylabel(measurement)
    pyplot.legend()
    pyplot.show()
    return

def splitplotall(dates, datesLabel, data, dataColor, dataLabel, predictions1, predictions1Color, predictions1Label, predictions2, predictions2Color, predictions2Label, predictions3, predictions3Color, predictions3Label, predictions4, predictions4Color, predictions4Label, split, measurement):

    test, testdates = splitcalculator(dates, data, split)

    pyplot.figure('Forecast')

    if is_date(dates[0]):
        pyplot.subplot(411)
        pyplot.plot([datetime.datetime.strptime(d,'%d/%m/%Y').date() for d in testdates], test, color = dataColor, label = dataLabel, linewidth = 2.0)
        pyplot.plot([datetime.datetime.strptime(d,'%d/%m/%Y').date() for d in testdates], predictions1, color = predictions1Color, label = predictions1Label, linewidth = 1)
        pyplot.xlabel(datesLabel)
        pyplot.ylabel(measurement)
        pyplot.legend(loc = "lower left")

        pyplot.subplot(412)
        pyplot.plot([datetime.datetime.strptime(d,'%d/%m/%Y').date() for d in testdates], test, color = dataColor, label = dataLabel, linewidth = 2.0)
        pyplot.plot([datetime.datetime.strptime(d,'%d/%m/%Y').date() for d in testdates], predictions2, color = predictions2Color, label = predictions2Label, linewidth = 1)
        pyplot.xlabel(datesLabel)
        pyplot.ylabel(measurement)
        pyplot.legend(loc = "lower left")

        pyplot.subplot(413)
        pyplot.plot([datetime.datetime.strptime(d,'%d/%m/%Y').date() for d in testdates], test, color = dataColor, label = dataLabel, linewidth = 2.0)
        pyplot.plot([datetime.datetime.strptime(d,'%d/%m/%Y').date() for d in testdates], predictions3, color = predictions3Color, label = predictions3Label, linewidth = 1)
        pyplot.xlabel(datesLabel)
        pyplot.ylabel(measurement)
        pyplot.legend(loc = "lower left")

        pyplot.subplot(414)
        pyplot.plot([datetime.datetime.strptime(d,'%d/%m/%Y').date() for d in testdates], test, color = dataColor, label = dataLabel, linewidth = 2.0)
        pyplot.plot([datetime.datetime.strptime(d,'%d/%m/%Y').date() for d in testdates], predictions4, color = predictions4Color, label = predictions4Label, linewidth = 1.4)
        pyplot.xlabel(datesLabel)
        pyplot.ylabel(measurement)
        pyplot.legend(loc = "lower left")

    else:
        pyplot.subplot(411)
        pyplot.plot(testdates, test, color = dataColor, label = dataLabel, linewidth = 2.0)
        pyplot.plot(testdates, predictions1, color = predictions1Color, label = predictions1Label, linewidth = 1)
        pyplot.xlabel(datesLabel)
        pyplot.ylabel(measurement)
        pyplot.legend(loc = "lower left")

        pyplot.subplot(412)
        pyplot.plot(testdates, test, color = dataColor, label = dataLabel, linewidth = 2.0)
        pyplot.plot(testdates, predictions2, color = predictions2Color, label = predictions2Label, linewidth = 1)
        pyplot.xlabel(datesLabel)
        pyplot.ylabel(measurement)
        pyplot.legend(loc = "lower left")

        pyplot.subplot(413)
        pyplot.plot(testdates, test, color = dataColor, label = dataLabel, linewidth = 2.0)
        pyplot.plot(testdates, predictions3, color = predictions3Color, label = predictions3Label, linewidth = 1)
        pyplot.xlabel(datesLabel)
        pyplot.ylabel(measurement)
        pyplot.legend(loc = "lower left")

        pyplot.subplot(414)
        pyplot.plot(testdates, test, color = dataColor, label = dataLabel, linewidth = 2.0)
        pyplot.plot(testdates, predictions4, color = predictions4Color, label = predictions4Label, linewidth = 1.4)
        pyplot.xlabel(datesLabel)
        pyplot.ylabel(measurement)
        pyplot.legend(loc = "lower left")

    pyplot.show()

    return
