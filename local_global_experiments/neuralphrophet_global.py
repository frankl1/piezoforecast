import argparse
import time
import pandas as pd
import numpy as np

from neuralprophet import NeuralProphet, set_log_level

set_log_level('ERROR')

parser = argparse.ArgumentParser()

parser.add_argument('dataset', type=str, help='the dataset file')
parser.add_argument('out_file', type=str, help="the result file")
parser.add_argument('-l', '--history', type=int, help='Length of the history', default=100)
parser.add_argument('-H', '--horizon', type=int, help='Length of the horizon', default=93)

args = parser.parse_args()
print("\n\n", args, "\n\n")

print("Running with the following parameters")
print(args)

dataset = args.dataset
out_file = args.out_file
horizon = args.horizon
history = args.history

assert history > horizon, "History must be greater than horizon"

def fit_and_predict(model, train_dict_df, test_dict_df, horizon):
    """This function trains the input model on the input on data
    Args:
    train_dict_df: a dictionary of dataframes with the columns ds (datetime) and y (time series values), and eventually lagged regressors
    test_dict_df: a dictionary of dataframes with the columns ds (datetime) and y (time series values), and eventually lagged regressors
    horizon: the horizon in the test data. 

    Return:
    a tuple containing in order: the train rmse, the test rmse, the train rmsse, the test rmsse, the training time, [h_1,h_2,...,h_horizon]
    """

    start_time = time.time()

    model.fit(train_dict_df, freq='D')

    learning_time = time.time() - start_time

    prediction_dict = model.predict(test_dict_df)

    rmsse_test_dict, rmsse_train_dict, rmse_test_dict, rmse_train_dict, out_prediction_dict = {}, {}, {}, {}, {}
    for k in test_dict_df:
        rmse_train = model.test(train_dict_df[k])['RMSE'][0]
        
        y_pred = prediction_dict[k].yhat1[-horizon:].values.astype(np.float32)
        y_true = test_dict_df[k].y[-horizon:].values.astype(np.float32)

        rmse_test = np.sqrt(np.mean((y_pred - y_true)**2))

        scaled_factor = np.sqrt(np.mean(train_dict_df[k].y.diff().dropna()**2))
        
        rmsse_train = rmse_train / scaled_factor
        rmsse_test = rmse_test / scaled_factor

        rmsse_train_dict[k] = rmsse_train
        rmsse_test_dict[k] = rmsse_test

        rmse_train_dict[k] = rmse_train
        rmse_test_dict[k] = rmse_test

        out_prediction_dict[k] = y_pred

    return rmse_train_dict, rmse_test_dict, rmsse_train_dict, rmsse_test_dict, learning_time, out_prediction_dict

all_data = pd.read_csv(dataset, delimiter=',', index_col=[0], parse_dates=['time'])
all_data.set_index('bss', drop=True, inplace=True)
all_data = all_data[all_data.time<"2021-01-16"]

# Get the piezo list, these piezos do not contains contant values
piezos = pd.read_csv('results/all-performance.csv', usecols=[0], squeeze=True)

def make_dataset(covariates):
    """This function creates the training and test time series dictionaries for global training with the input covariate"""
    df_train_dict = {}
    df_test_dict = {}
    for bss in piezos:
        aux = all_data.loc[bss].copy()

        tmp = pd.DataFrame({'ds': aux.time[:-horizon], 'y': aux.p[:-horizon].astype(np.float32)})
        for cov in covariates:
            tmp[cov] = aux[cov][:-horizon].astype(np.float32).interpolate(method='linear')

        df_train_dict[bss] = tmp
        
        # create a test set dataset
        tmp = pd.DataFrame({'ds': aux.time, 'y': aux.p.astype(np.float32)})
        for cov in covariates:
            tmp[cov] = aux[cov].astype(np.float32).interpolate(method='linear')

        df_test_dict[bss] = tmp

    return df_train_dict, df_test_dict
    
result_list = []
for cov_list in [['tp', 'e'], ['tp'], ['e'], []]:
    print("\n\n++++++++++++++Running with covariate", cov_list)

    df_train_dict, df_test_dict = make_dataset(cov_list)

    model =  NeuralProphet(global_normalization=True) if len(cov_list) == 0 else NeuralProphet(n_lags=horizon, n_forecasts=1, global_normalization=True)
    for cov in cov_list:
        model = model.add_lagged_regressor(cov)
    
    rmse_train_dict, rmse_test_dict, rmsse_train_dict, rmsse_test_dict, learning_time, predictions_dict = fit_and_predict(model, df_train_dict, df_test_dict, horizon)

    df_pred = pd.DataFrame(predictions_dict, index=[f'h{i}' for i in range(1, horizon+1)]).T

    res = pd.DataFrame({
        'model': 'NeuralProphet',
        'rmse_train': rmse_train_dict,
        'rmse_test': rmse_test_dict,
        'rmsse_train': rmsse_train_dict,
        'rmsse_test': rmsse_test_dict,
        'learningtime': learning_time,
        'use_exo_rain': 'tp' in cov_list,
        'use_exo_evo': 'e' in cov_list,
    }).join(df_pred)

    result_list.append(res)

    pd.concat(result_list).to_csv(out_file) # save at each iteration juste in case an error occurs
