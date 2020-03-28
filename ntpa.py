#
# ntpa.py
# Network Traffic Prediction Application | Version 4.4.42
#
# Created by Anestis Dalgkitsis in 15/02/2017.
# Copyright 2017 Anestis Dalgkitsis. All rights reserved.
#
# Contact: anestisdalgkitsis@gmail.com
#

# Global Depentencies

import argparse
import warnings

# Silence Warnings

warnings.simplefilter(action = 'ignore', category = FutureWarning)

# Define Terminal Arguments

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", dest = "inputfile", default = "ltedata.csv", help="Input File Path. Files can only be .csv.")
parser.add_argument("-f", "--forecast", dest = "forecasttype", default = "EVALUATION", help="Can be EVALUATION, FORECAST or NEXTDAY")
parser.add_argument("-c", "--cell", dest = "cellid", default = "301517", help="Cell ID. Default is the first in the input file.")
parser.add_argument("-a", "--algorithm", dest = "algorithm", default = "ALL", help="Use LSTM, SLSTM, RBF, ARIMA or ALL for comparison. DATASETINFO or BESTEXPRESS to invoke data utilities.")
parser.add_argument("-m", "--measurement", dest = "measurement", default = "THROUGHPUT", help="Can be THROUGHPUT, DOWNLOADTRAFFIC or ACTIVITYTIME.")
parser.add_argument("-n", "--normalization", dest = "normalization", default = "FALSE", help="Can be EXCESSAVERAGE.")
args = parser.parse_args()

# Local Depentencies

import load
import plot

import filters
import datasetinfo

import rbf
#import lstm # Lazy Load for better performance
import arima
import sarimax

# --------------------------------------------------

# Global Settings
# /!\ Defaults for optimal performance. Modify at your own risk

# LSTM NETWORK MODEL SETTINGS

batchsize = 1 # Increases fitting time drastically. You should only pass inputs with a number of samples that can be divided by the batch size.
epochs = 17 # Keep epochs low to prevent overfitting or use dropout.
neurons = 2 # Keep between 2 and 5, more neurons increase fitting time.

# Smart LSTM SETTINGS
# /!\ WARNING Smart LSTM shares network model settings with LSTM from above.

blocksNumber = 5 # Increase or decrease according to the size of the dataset.

# ARIMA SETTINGS -> ARIMA(p,d,q)

lag = 5 # p Vaule: Lag value for autoregression. Less is faster, more is accurate.
difference = 1 # d Value: Difference order of 1 makes the time series stationary.
model = 0 # q Value: Moving Average Model. Default model for ARIMA is 0.

# RBF SETTINGS

C = 1e3 # Penalty parameter C of the error term.
gamma = 0.1 # Kernel coefficient for RBF. If gamma is ‘auto’ then 1/n will be used instead.

# GENERAL SETTINGS

split = 0.6 # Amount of dataset split for FORECAST. The dataset will be trained in "split * 100 %" amount of the dataset and will forecast the remaining.

# --------------------------------------------------

# Fetch Data

dates, intdates, throughputs, downloadtraffic, activitytime = load.load(args.inputfile, args.cellid)

# Data Routing

if args.measurement == 'THROUGHPUT':
    measurement = throughputs
    filteredmeasurement = throughputs

elif args.measurement == 'DOWNLOADTRAFFIC':
    measurement = downloadtraffic
    filteredmeasurement = downloadtraffic

elif args.measurement == 'ACTIVITYTIME':
    measurement = activitytime
    filteredmeasurement = activitytime

# Filtering

if args.normalization == 'EXCESSAVERAGE':
    filteredmeasurement = filters.excessaverage(measurement[:], excess = 0.8)

# Filtered Data Routing

if args.algorithm == 'ALL':

    if args.forecasttype == 'EVALUATION':
        print(" [RBF]")
        rbfpredictions, rbferror, rbfduration = rbfpredictions, rbferror, duration = rbf.rbf(intdates, filteredmeasurement, measurement)
        print(" [ARIMA]")
        arimapredictions, arimaerror, arimaduration = arimapredictions, arimaerror, duration = arima.arima(intdates, filteredmeasurement, measurement, lag, difference, model)
        print(" [SARIMAX]")
        sarimaxpredictions, sarimaxerror, sarimaxduration = sarimax.sarimax(intdates, filteredmeasurement, measurement)
        print(" [LSTM]")
        import lstm
        print(" [i] TensorFlow Library Loaded")
        lstmpredictions, lstmerror, totalduration, fitduration, lstmpredictionduration = lstm.evaluationlstm(filteredmeasurement, measurement, batchsize, epochs, neurons)
        print(" [i] Evaluation Report")
        print('  |- RBF     MSE :', rbferror,'@', rbfduration, '(s)')
        print('  |- ARIMA   MSE :', arimaerror,'@', arimaduration, '(s)')
        print('  |- SARIMAX MSE :', sarimaxerror,'@', sarimaxduration, '(s)')
        print('  |- LSTM    MSE :', lstmerror,'@', lstmpredictionduration, '(s)')
        plot.plotall(dates[0:len(dates) - 1], 'Days', measurement[0:len(measurement) - 1], 'black', 'Data', rbfpredictions[0:len(rbfpredictions) - 1], 'red', 'RBF', arimapredictions[0:len(arimapredictions) - 1], 'blue', 'ARIMA', sarimaxpredictions, 'green', 'SARIMAX', lstmpredictions, 'orange', 'LSTM', args.measurement)

    elif args.forecasttype == 'FORECAST':
        print(" [RBF]")
        rbfpredictions, rbferror, rbfduration = rbfpredictions, error, duration = rbf.forecastrbf(intdates, filteredmeasurement, measurement, split)
        print(" [ARIMA]")
        arimapredictions, arimaerror, arimaduration = arimapredictions, error, duration = arima.forecastarima(intdates, filteredmeasurement, measurement, split, lag, difference, model)
        print(" [SARIMAX]")
        sarimaxpredictions, sarimaxerror, sarimaxduration = sarimaxpredictions, error, duration = sarimax.forecastsarimax(intdates, filteredmeasurement, measurement, split)
        print(" [LSTM]")
        import lstm
        print(" [i] TensorFlow Library Initialized")
        lstmpredictions, lstmerror, lstmtotalduration, lstmfitduration, lstmpredictionduration = lstmpredictions, error, totalduration, fitduration, predictionduration = lstm.forecastlstm(filteredmeasurement, measurement, split,  batchsize, epochs, neurons)
        print(" [i] Forecast Report")
        print('  |- RBF     MSE :', rbferror,'@', rbfduration, '(s)')
        print('  |- ARIMA   MSE :', arimaerror,'@', arimaduration, '(s)')
        print('  |- SARIMAX MSE :', sarimaxerror,'@', sarimaxduration, '(s)')
        print('  |- LSTM    MSE :', lstmerror,'@', lstmpredictionduration, '(s)')
        plot.splitplotall(dates[0:len(dates) - 1], 'Days', measurement, 'black', 'Data', rbfpredictions[0:len(rbfpredictions) - 1], 'red', 'RBF', arimapredictions[0:len(arimapredictions) - 1], 'blue', 'ARIMA', sarimaxpredictions, 'green', 'SARIMAX', lstmpredictions, 'orange', 'LSTM', split, args.measurement)

elif args.algorithm == 'LSTM':

    print(" [LSTM]")

    import lstm

    print(" [i] TensorFlow Library Initialized")

    if args.forecasttype == 'EVALUATION':
        predictions, error, totalduration, fitduration, predictionduration = lstm.evaluationlstm(filteredmeasurement, measurement, batchsize, epochs, neurons)
        plot.plot(dates[0:len(dates) - 1], 'Days', measurement[0:len(measurement) - 1], 'black', 'Data', predictions, 'orange', 'LSTM', args.measurement)

    elif args.forecasttype == 'NEXTDAY':
        print(lstm.nextday(intdates, filteredmeasurement, len(intdates) + 1, batchsize, epochs, neurons))

    elif args.forecasttype == 'FORECAST':
        predictions, error, totalduration, fitduration, predictionduration = lstm.forecastlstm(filteredmeasurement, measurement, split,  batchsize, epochs, neurons)
        plot.splitplot(dates[0:len(dates) - 1], 'Days', measurement, 'black', 'Data', predictions, 'orange', 'LSTM', split,  args.measurement)

elif args.algorithm == 'SLSTM':

    print(" [Smart LSTM]")

    import lstm

    print(" [i] TensorFlow Library Initialized")

    if args.forecasttype == 'FORECAST':
        predictions, error, totalduration, fitduration, predictionduration = lstm.slstm(filteredmeasurement, measurement, split, blocksNumber,  batchsize, epochs, neurons)
        plot.splitplot(dates[0:len(dates) - 1], 'Days', measurement, 'black', 'Data', predictions, 'magenta', 'Smart LSTM', split,  args.measurement)

elif args.algorithm == 'RBF':

    print(" [RBF]")

    if args.forecasttype == 'EVALUATION':
        predictions, error, duration = rbf.rbf(intdates, filteredmeasurement, measurement)
        plot.plot(dates, 'Days', measurement, 'black', 'Data', predictions, 'red', 'RBF', args.measurement)

    elif args.forecasttype == 'NEXTDAY':
        print(rbf.forecast(intdates, filteredmeasurement, len(intdates) + 1))

    elif args.forecasttype == 'FORECAST':
        predictions, error, duration = rbf.forecastrbf(intdates, filteredmeasurement, measurement, split)
        plot.splitplot(dates, 'Days', measurement, 'black', 'Data', predictions, 'red', 'RBF', split, args.measurement)

elif args.algorithm == 'ARIMA':

    print(" [ARIMA]")

    if args.forecasttype == 'EVALUATION':
        predictions, error, duration = arima.arima(intdates, filteredmeasurement, measurement, lag, difference, model)
        plot.plot(dates, 'Days', measurement, 'black', 'Data', predictions, 'blue', 'ARIMA', args.measurement)

    elif args.forecasttype == 'NEXTDAY':
        print(arima.nextday(intdates, filteredmeasurement, len(intdates) + 1, lag, difference, model))

    elif args.forecasttype == 'FORECAST':
        predictions, error, duration = arima.forecastarima(intdates, filteredmeasurement, measurement, split, lag, difference, model)
        plot.splitplot(dates, 'Days', measurement, 'black', 'Data', predictions, 'blue', 'ARIMA', split, args.measurement)

elif args.algorithm == 'SARIMAX':

    print(" [SARIMAX]")

    if args.forecasttype == 'EVALUATION':
        predictions, error, duration = sarimax.sarimax(intdates, filteredmeasurement, measurement)
        plot.plot(dates[0:len(dates) - 1], 'Days', measurement[0:len(dates) - 1], 'black', 'Data', predictions, 'green', 'SARIMAX', args.measurement)

    elif args.forecasttype == 'NEXTDAY':
        print(sarimax.nextday(intdates, filteredmeasurement, len(intdates)))

    elif args.forecasttype == 'FORECAST':
        predictions, error, duration = sarimax.forecastsarimax(intdates, filteredmeasurement, measurement, split)
        plot.splitplot(dates[0:len(dates) - 1], 'Days', measurement, 'black', 'DATA', predictions, 'green', 'SARIMAX', split, args.measurement)

elif args.algorithm == 'DATASETINFO':
    datasetinfo.miner(dates, measurement)

elif args.algorithm == 'BESTEXPRESS':
    print(" [Best Express]")
    print(" [x] Deleting Dataset")
    print(" [i] Calculate entire dataset size")
    cellids = load.alluniquecellids(args.inputfile)

    print(" [i] Initializing Smart LSTM")
    import lstm
    print(" [i] TensorFlow Library Initialized")

    print(" [i] Initializing Forecasting Emulator")
    errors = []
    cells = []

    print(" [>] Running Forecasting Emulator X")
    for i in range(40): #len(cellids)
        print('  |- Progress : %s / %s' % (i, len(cellids)))
        print('  |- Cell ID : %s' % (cellids[i]))
        print ("Percentage",i/len(cellids)*100," Percent Complete         \r")
        datesX, intdatesX, throughputsX, downloadtrafficX, activitytimeX = load.load(args.inputfile, cellids[i])
        if len(throughputsX) == 0:
            print("EMPTY")
        predictionsX, errorX, totaldurationX, fitdurationX, predictiondurationX = lstm.slstm(throughputsX, throughputsX, split, blocksNumber,  batchsize, epochs, neurons)
        cells.append(cellids[i])
        errors.append(errorX)
    print(" [v] Emulation Finished Successfully!")

    print(" [i] Ranking Results")
    for c in range(40-1,0,-1):#len(cellids)
        for i in range(c):
            if errors[i] > errors[i+1]:
                # Sort Error
                temp = errors[i]
                errors[i] = errors[i+1]
                errors[i+1] = temp
                # Sort Cells
                temp2 = cells[i]
                cells[i] = cells[i+1]
                cells[i+1] = temp2

    print(" [i] Lowest Forecasted MSE Cells")
    for i in range(40):#len(cellids)
        print('  |- Rank : #%s | Cell ID : %s | MSE : %s' % (i, cells[i], errors[i]))

# ASCII Epilogue

print("anestisdalgkitsis ×")
