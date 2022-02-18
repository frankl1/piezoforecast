# Défi EGC 2022 : prévoir l’évolution du niveau de nos nappes phréatiques

https://www.egc.asso.fr/manifestations/defi-egc/defi-egc-2022-prevoir-levolution-du-niveau-de-nos-nappes-phreatiques.html

## Data preprocessing
1. [x] Fill missing values
2. [x] Data scalling
3. [x] New features
4. [x] starting date


## Models
1. [x]Baselines: le next predictions are the current observed values
2. [x] CNN and LSTM from Lara-Benítez, P., Carranza-García, M., & Riquelme, J. C. (2021). An Experimental Review on Deep Learning Architectures for Time Series Forecasting. arXiv preprint arXiv:2103.12057.
3. [x] Optimize models architectures
    1. [x] Add residual connections
    2. [x] Add batch normalization


## Training
1. [x] create training set, validation set and test set (CV for time series forecasting)
2. train each of the selected models and keep the best
3. [x] add model checkpoint
4. [x] shuffle data for non-RNN models


## Live validation
1. [x] Forecast the comming week
2. [x] Validate the forecasting one week after
    1. Create a function for live validation
3. [x] Record statistics

## Final forcast
1. Take the best model regarding the live forecast
2. For each dataset
    1. Train the model
    2. Forecast the $93$ values from 15 Oct 2021 to 15 Jan 2022 for each dataset
    3. Record the forcasts
3. Merge every forecasts in a file and submit



