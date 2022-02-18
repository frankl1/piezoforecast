import os
import argparse
import time
import logging
import pandas as pd
import numpy as np
import tensorflow as tf

from prophet import Prophet
from neuralprophet import NeuralProphet

PROPHET_PREDICTOR = 'prophet'
NEURAL_PROPHET_PREDICTOR = 'neuralprophet'

parser = argparse.ArgumentParser()

parser.add_argument('dataset', type=str, help='the dataset file')
parser.add_argument('out_file', type=str, help="the result file")
parser.add_argument('-H', '--horizon', type=int, help='Length of the horizon', default=93)
parser.add_argument('-z', '--znormalize', action='store_true', help="If set, z-normalize the time series")
parser.add_argument('-p', '--predictor', type=str, choices=(PROPHET_PREDICTOR, NEURAL_PROPHET_PREDICTOR), default=PROPHET_PREDICTOR)

args = parser.parse_args()
print("\n\n", args, "\n\n")

print("Running with the following parameters")
print(args)

dataset = args.dataset
out_file = args.out_file
znormalize = args.znormalize
predictor = args.predictor
horizon = args.horizon

columns = ['bss_code','model','rmse_train','rmse_test','rmsse_train','rmsse_test','learningtime', 'use_exo_rain', 'use_exo_evo'] + [f'h{i}' for i in range(1, 94)]

def fit_and_predict(model, data, horizon):
    """This function trains the input model on the input on data
    Args:
    data: a dataframe with the columns ds (datetime) and y (time series values)
    horizon: a integer represanting the horizon length

    Return:
    a tuple containing in order: the train rmse, the test rmse, the train rmsse, the test rmsse, the training time, [h_1,h_2,...,h_horizon]
    """

    start_time = time.time()

    if isinstance(model, Prophet):
        model.fit(data[:-horizon])
    else:
        model.fit(data[:-horizon], freq='D')

    learning_time = time.time() - start_time

    predictions = model.predict(data)

    if isinstance(model, Prophet):
        predictions = predictions.yhat
    else :
        predictions = predictions.yhat1.astype(float)

    mse_train = tf.keras.metrics.mean_squared_error(predictions[:-horizon], data[:-horizon].y).numpy()
    mse_test = tf.keras.metrics.mean_squared_error(predictions[-horizon:], data[-horizon:].y).numpy()

    scaled_factor = np.mean(data.y.diff().dropna()**2)
    msse_train = mse_train / scaled_factor
    msse_test = mse_test / scaled_factor

    return np.sqrt(mse_train), np.sqrt(mse_test), np.sqrt(msse_train), np.sqrt(msse_test), learning_time, predictions[-horizon:]

all_data = pd.read_csv(dataset, delimiter=',', index_col=[0], parse_dates=['time'])
all_data.set_index('bss', drop=True, inplace=True)

to_skip = []
if os.path.isfile(out_file):
    to_skip = pd.read_csv(out_file, usecols=['model', 'bss_code'])
    to_skip = (to_skip.bss_code + to_skip.model).values

with open(out_file, 'a+', buffering=1) as f:

    if os.stat(out_file).st_size == 0:
        f.write(','.join(columns) + '\n')

    count = 0

    for bss in all_data.index.unique():
        count += 1

        if bss + predictor in to_skip:
            continue

        print('\n\n**************-> Executing on piezo No:', count, 'bss:', bss, '\n\n')

        data = all_data.loc[bss].sort_index()

        data.e = data.e.interpolate(method='linear')
        data.tp = data.tp.interpolate(method='linear')

        data = data.rename(columns={'time': 'ds', 'p': 'y'})

        y_mean, y_std = None, None
        if znormalize:
            y_mean = data.y.mean()
            y_std = data.y.std()

            data.y = (data.y - y_mean) / y_std
        
        model = None 
        lines = ""
        for cov_list in [[], ['tp'], ['e'], ['tp', 'e']]:
            if predictor == PROPHET_PREDICTOR:
                model = Prophet()
                for cov in cov_list:
                    model.add_regressor(cov)
            else:
                model =  NeuralProphet() if len(cov_list) == 0 else NeuralProphet(n_lags=5, n_forecasts=1)
                for cov in cov_list:
                    model.add_lagged_regressor(cov)

            rmse_train, rmse_test, rmsse_train, rmsse_test, learning_time, predictions = fit_and_predict(model, data[['ds', 'y']+cov_list], horizon)

            if znormalize:
                predictions = predictions * y_std + y_mean
                rmse_train = rmse_train * y_std + y_mean
                rmse_test = rmse_test * y_std + y_mean
                rmsse_train = rmsse_train * y_std + y_mean
                rmsse_test = rmsse_test * y_std + y_mean

            lines += ','.join(np.array([bss, predictor, rmse_train, rmse_test, rmsse_train, rmsse_test, learning_time, 'tp' in cov_list, 'e' in cov_list] + predictions.tolist(), dtype=str))+'\n'
        
        f.write(lines)