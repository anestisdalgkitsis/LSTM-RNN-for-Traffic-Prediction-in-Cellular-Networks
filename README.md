# Long Short-Term Memory Recurrent Neural Network for Traffic Prediction in Cellular Networks.

We designed and implemented a Neural Network that can identify recurrent patterns in various metrics used in cellular network traffic forecasting and conduct sufficient conclusions about the traffic in the future. 
Thanks to custom architecture and memory, this Neural Network can handle prediction faster and even more accurate in real life scenarios.
This proposal may offer a solution for service providers to enhance cellular network performance, by utilizing effectively all available resources with smart strategic planning, that uses predictions by this proposed Neural Network architecture.
Multiple predictions were made in the same data-set, in order to provide a robust conclusion about the performance and precision of the proposed Neural Network against other algorithms from similar literature.

## Documents

- Paper:  http://users.uowm.gr/louta/CONFERENCES/C55.pdf
- Master Thesis: https://dspace.uowm.gr/xmlui/bitstream/handle/123456789/1529/Anestis_Dalgkitsis_Thesis.pdf

## How to use

Locate the project folder in your terminal and run the following command:
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

## Disclaimer about the project

- This project is not maintained any more.
- Dataset is not provided for privacy reasons. Instead a dummy dataset can be found at ltedata.csv.
- Not all parts of the code are included in this repository for privacy reasons.
- Keep in mind that this is thesis project.
