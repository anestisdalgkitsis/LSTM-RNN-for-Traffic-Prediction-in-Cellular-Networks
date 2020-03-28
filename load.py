#
# load.py
# Extented Data Loading Functions | Version 3.4.3
#
# Created by Anestis Dalgkitsis on 02/10/2017.
# Copyright 2017 Anestis Dalgkitsis. All rights reserved.
#
# Contact: anestisdalgkitsis@gmail.com
#

# Depentencies

import csv

from datetime import datetime

# Global Tables

dates = []
intdates = []
throughputs = []
downloadtraffic = []
activitytime = []

# Functions

def countdays(dates):
    for i in range(0, len(dates)):
        intdates.append(i)
    return

def floatconvert(measurement):
    for i in range(0, len(measurement)):
        measurement.append(float(measurement[i]))
        measurement.pop(0)
    return

def alluniquecellids(filename):
    cells = []
    setcells = []
    with open(filename, 'r') as f:
        data = csv.reader(f, delimiter=";", quotechar='|')
        next(f)
        for row in data:
            cells.append(row[4])
        setcells = list(set(cells))
        return setcells

def load(filename, cellid):

    del dates[:]
    del intdates[:]
    del throughputs[:]
    del downloadtraffic[:]
    del activitytime[:]

    with open(filename, 'r') as f:
        data = csv.reader(f, delimiter=";", quotechar='|')

        for row in data:
            if cellid == row[4]:
                dates.append(row[0])
                downloadtraffic.append(row[1])
                activitytime.append(row[2])
                throughputs.append(row[3].replace(',','.'))

    countdays(dates)
    floatconvert(downloadtraffic)
    floatconvert(activitytime)
    floatconvert(throughputs)

    print(" [i] Dataset Imported")

    return dates, intdates, throughputs, downloadtraffic, activitytime
