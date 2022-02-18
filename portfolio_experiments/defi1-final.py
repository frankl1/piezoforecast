#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import requests
import os
import seaborn as sns
import numpy as np
import tensorflow as tf
import IPython
import argparse
import time

from niveau_nappe_core import *

global_start_time = time.time()

parser = argparse.ArgumentParser()

parser.add_argument('-e', '--max_epoch', type=int, help='Maximum number of epochs', default=50)
parser.add_argument('-b', '--batch_size', type=int, help='Batch size', default=64)
parser.add_argument('-p', '--patience', type=int, help='The patience value for early stopping callback', default=5)
parser.add_argument('-v', '--verbose', type=int, help='The verbosing level for keras', default=2)

args = parser.parse_args()

# The horizon size
label_width = 93

# history_factor
h_factor = 1.25

# An history length of 
input_width = int(label_width * h_factor)

# Number of feature values to predict at once
shift = label_width

# batch size
batch_size = args.batch_size

MAX_EPOCHS = args.max_epoch

patience = args.patience

verbose = args.verbose


#sources = pd.read_csv('../data_collection/dataset_stations.csv', delimiter=',', index_col=0)

all_data = pd.read_csv('../data_collection/dataset_nomissing_linear.csv', delimiter=',', index_col=['bss'], parse_dates=['time'])
sources = all_data.index

with open('performances.csv', 'a+', buffering=1) as perf_file:

    if not perf_file.tell():
        perf_file.write(',val_loss,test_loss,best_model_at_test,runtime_in_sec\n')
    else:
        perf = pd.read_csv('performances.csv', index_col=0)
        sources = sources[~sources.isin(perf.index)]

    for code_bss in sources:
        try:
            print('\n\nRunning on bss_id:', code_bss)

            start_time = time.time()

            # # Retrieve data
            if code_bss not in all_data.index:
                continue
                
            data = all_data.loc[code_bss].set_index('time')
            data = data.sort_index()
            
            # select depth col
            data = data[['p']]
            
            # dropna
            data = data.dropna()

            # # Preprocessing

            # ## Resampling the data to have a measure per day

            start_date = data.index.min()

            data = data.loc[start_date:, :]

            new_index = pd.date_range(start_date, data.index.max())

            missing_index = new_index.difference(data.index)

            print("There are", missing_index.size, "missing dates over ", new_index.size, " => ", round(100 * missing_index.size/new_index.size, 2), "% of data")

            # Adding date features
            data = add_date_features(data)

            if missing_index.size > 0:
                # Filling missing observations with Gaussian Processes

                imputer = build_imputer_model()

                imputer.fit(data[['year', 'month', 'quarter', 'weekday', 'day']], data['p'])

                print('\n\nimputer.Score:', imputer.score(data[['year', 'month', 'quarter', 'weekday', 'day']], data['p']), '\n\n')


                # Predict missing values
                missing_X = np.concatenate([missing_index.year.values.reshape((-1, 1)), 
                                    missing_index.month.values.reshape((-1, 1)), 
                                    missing_index.quarter.values.reshape((-1, 1)), 
                                    missing_index.weekday.values.reshape((-1, 1)), 
                                    missing_index.day.values.reshape((-1, 1))], axis=1)
                missing_X
                missing_y = imputer.predict(missing_X)
                
                cols = ['year', 'month', 'quarter', 'weekday', 'day', 'p']

                missing_X_y = pd.DataFrame(data=np.concatenate([missing_X, np.expand_dims(missing_y, axis=1)], axis=1), 
                                        columns=cols, 
                                        index=missing_index)

                data_no_missing = data[cols].append(missing_X_y)
                data_no_missing = data_no_missing.sort_index()
            else:
                data_no_missing = data

            # ### Feature engineering
            # 
            # - Add temperature
            # - Add precipitation

            # First and second degree diff
            data_no_missing = add_derivate_features(data_no_missing, 'p')

            # ### Split dataset 
            # 
            # $70\%$ for training, $20\%$ for validation and $10\%$ for test

            # In[14]:

            features = ['p', 'year', 'month', 'quarter', 'weekday', 'day', 'diff_p', 'diff30_p', 'diff90_p', 'diff180_p', 'diff_diff_p']

            df = data_no_missing[features]

            n_features = len(features)

            n = df.shape[0]

            len_train = int(n * 0.7)
            len_val = int(n * 0.2)

            train_df = df[:len_train]
            val_df = df[len_train:len_train+len_val]
            test_df = df[len_train+len_val:]

            print('\tTrain shape:', train_df.shape)
            print('\tVal shape:', val_df.shape)
            print('\tTest shape:', test_df.shape)
            print('\tNumber of observations:', n) 
            print('\tFeatures:', features, 'number:', n_features)


            # ## Data normalization
            # 
            # To do when working with more than one feature


            # In[16]:


            train_mean = train_df.mean()
            train_std = train_df.std()

            train_df = (train_df - train_mean) / train_std
            val_df = (val_df - train_mean) / train_std
            test_df = (test_df - train_mean) / train_std

            # ### Data windowing

            window = WindowGenerator(input_width=input_width, 
                                    label_width=label_width, 
                                    shift=shift,
                                    batch_size=batch_size,
                                    train_df=train_df, val_df=val_df, test_df=test_df, label_columns=['p'])

            no_suffled_window = WindowGenerator(input_width=input_width, 
                                    label_width=label_width, 
                                    shift=shift,
                                    batch_size=batch_size,
                                    train_df=train_df, val_df=val_df, test_df=test_df, label_columns=['p'], shuffle=False)

            # ## Model 

            models = {}

            lstm = tf.keras.models.Sequential([
                # Shape [batch, time, features] => [batch, time, lstm_units]
                tf.keras.layers.LSTM(128, return_sequences=True),
                tf.keras.layers.LSTM(32, return_sequences=False),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(units=label_width),
                tf.keras.layers.Reshape((label_width, -1))
            ], name='lstm')

            conv = tf.keras.Sequential([
                tf.keras.layers.Conv1D(filters=256, kernel_size=7, activation='relu', padding='same'),
                tf.keras.layers.Conv1D(filters=128, kernel_size=7, activation='relu', padding='same'),
                tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu', padding='same'),
                tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
                tf.keras.layers.Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(units=label_width),
                tf.keras.layers.Reshape((label_width, -1))
            ], name='conv')

            CONV_WIDTH = 3
            filters = 64
            n_blocks = 2
            use_bn = True

            rnForecaster = RNForecaster(kernel_size=CONV_WIDTH, label_width=label_width, filters=filters, n_blocks=n_blocks, use_batch_norm=use_bn)

            models['lstm'] = lstm
            models['conv'] = conv
            models['Resnet'] = rnForecaster


            best_model = None 
            best_test_score = np.inf 
            best_val_score = np.inf ## the validation score associated to the best test score

            for name, forecaster in models.items():
                window_tmp = no_suffled_window if name == 'lstm' else window

                history = compile_and_fit(forecaster, window_tmp, patience=patience, epochs=MAX_EPOCHS, verbose=verbose)

                val_perf = forecaster.evaluate(window_tmp.val, verbose=verbose)
                test_perf = forecaster.evaluate(window_tmp.test, verbose=verbose)

                if test_perf < best_test_score:
                    best_model = name
                    best_test_score = test_perf
                    best_val_score = val_perf
                    label_index = window_tmp.column_indices['p'] if 'AR' in name else None

                models[name] = forecaster

            predictions = forecast(models[best_model], best_model, test_df, input_width, label_width, train_mean, train_std, label_index=label_index)
            
            predictions = pd.DataFrame({
                'CODE_BSS': code_bss,
                'DATE': predictions.index,
                'PROFONDEUR': predictions.p * train_std.p + train_mean.p
            })

            if os.path.exists('submission.csv'):
                submission = pd.read_csv('submission.csv')
                submission = submission.append(predictions)
            else:
                submission = predictions

            submission.to_csv('submission.csv', index=False)

            end_time = time.time()

            perf_file.write(f"{code_bss},{best_val_score},{best_test_score},{best_model},{end_time - start_time}\n")
        except Exception as e:
            print("Error on Piezo: ", code_bss)
            print(e)

global_run_time = time.time() - global_start_time
global_run_time = round(global_run_time / 3600, 2)

print(f'All done. Took {global_run_time} H :)')

