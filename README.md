# Long Short-Term Memory Recurrent Neural Network for Traffic Prediction in Cellular Networks.

We designed and implemented a Neural Network that can identify recurrent patterns in various metrics used in cellular network traffic forecasting and conduct sufficient conclusions about the traffic in the future. 
Thanks to custom architecture and memory, this Neural Network can handle prediction faster and even more accurate in real life scenarios.
This proposal may offer a solution for service providers to enhance cellular network performance, by utilizing effectively all available resources with smart strategic planning, that uses predictions by this proposed Neural Network architecture.
Multiple predictions were made in the same data-set, in order to provide a robust conclusion about the performance and precision of the proposed Neural Network against other algorithms from similar literature.

## Documents

- Paper:  http://users.uowm.gr/louta/CONFERENCES/C55.pdf
- Master Thesis: https://dspace.uowm.gr/xmlui/bitstream/handle/123456789/1529/Anestis_Dalgkitsis_Thesis.pdf

## How to use

Download, locate the project folder in your terminal and run the following command:
```
python ntpa.py -i ltedata.csv <<arguments>>
```
Where `<<arguments>>` are listed below. If no arguments given the application will run with the default settings.

### Arguments

- -i Inputfile (e.g. ltedata.csv).
- -f Forecast type (can be EVALUATION, FORECAST or NEXTDAY).
- -c Cell ID (e.g. 914753)
- -a Algorithm (can be ALL, LSTM, SLSTM, RBF or ARIMA).
- -m Measurement (can be THROUGHPUT, DOWNLOADTRAFFIC or ACTIVITYTIME)
- -n Normalization (can be FALSE or EXCESSAVERAGE).

### Dataset

Dataset must be a `.csv` file with the following structure:

```
DAYSAMPLED;DL_TRAFFIC_BITS;ACTIVITY_TIME_MSEC;CELL_THROUGHPUT_MBPS;Cell ID
01/01/2017;7448749488     ;550004            ;4      ,65          ;914753
...
```

- A dummy file `ltedata.csv` is included in the project for your convenience.
- Cell throughput branch contains both upload and download throughputs.

> Sensitive data are removed. This file does not contain real data. Please replace with your own, but keep the initial structure. 

## Disclaimer about the project

- This project is not maintained any more.
- Dataset is not provided for privacy reasons. Instead a dummy dataset can be found at ltedata.csv.
- Not all parts of the code are included in this repository for privacy reasons.
- Keep in mind that this is thesis project.

### Cite this work 

Bibtex:
```
@inproceedings{inproceedings,
author = {Dalgkitsis, Anestis and Louta, Malamati and Karetsos, George},
year = {2018},
month = {11},
pages = {28-33},
title = {Traffic forecasting in cellular networks using the LSTM RNN},
isbn = {978-1-4503-6610-6},
doi = {10.1145/3291533.3291540}
}
```

> Do you plan to extend or contribute to our work? If yes, it would be great to mention us in your work. Please keep us informed.

More information in ResearchGate website:
DOI: 10.1145/3291533.3291540
https://www.researchgate.net/publication/330226800_Traffic_forecasting_in_cellular_networks_using_the_LSTM_RNN
