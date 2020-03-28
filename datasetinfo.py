#
# datasetinfo.py
# Dataset Information | Version 1.8.2
#
# Created by Anestis Dalgkitsis in 19/01/2018.
# Copyright 2017 Anestis Dalgkitsis. All rights reserved.
#
# Contact: anestisdalgkitsis@gmail.com
#

# Depentencies

import math
import numpy
from scipy.signal import find_peaks_cwt
import datetime as datetime

# Utility Functions

#def dow(day):
#if intday == 0:

def dow(dayNumber):
    days = ["Monday   ", "Tuesday  ", "Wednesday", "Thursday ", "Friday   ", "Saturday ", "Sunday   "]
    return days[dayNumber]

# Functions

def miner(dates, dataset):
    print(" [DATASET STATISTICS REPORT]")
    print(" [i] Raw Data Information")
    weekendSum = 0
    weekendCounter = 0
    weekdaySum = 0
    weekdayCounter = 0
    for i in range(len(dataset)):
        print('  |- Date : %s | Day : %s | Value : %f' % (dates[i], dow([datetime.datetime.strptime(dates[i], '%d/%m/%Y').weekday()][0]), dataset[i]))
        if ([datetime.datetime.strptime(dates[i], '%d/%m/%Y').weekday()][0]) == 5 | ([datetime.datetime.strptime(dates[i], '%d/%m/%Y').weekday()][0] == 6):
            weekendSum = dataset[i] + weekendSum
            weekendCounter += 1
        else:
            weekdaySum = dataset[i] + weekdaySum
            weekdayCounter += 1
    weekendMean = weekendSum / weekendCounter
    weekdayMean = weekdaySum / weekdayCounter
    peaks = find_peaks_cwt(numpy.array(dataset), numpy.arange(1, 25))
    print(" [i] Dataset Seasonality")
    print('  |- Total Days             : ', len(dates))
    print('  |- Total Months           : ', math.floor(len(dates) / 30))
    print('  |- Total Mean Value       : ', (weekdayMean + weekendMean) / 2)
    print('  |- Total Weekdays         : ', weekdayCounter)
    print('  |- Weekday Mean Value     : ', weekdayMean)
    print('  |- Total Weekends         : ', weekendCounter)
    print('  |- Weekend Mean Value     : ', weekendMean)
    print('  |- Spikes Detected        : ', len(peaks))
