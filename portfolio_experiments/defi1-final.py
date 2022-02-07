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

parser.add_argument('-e', '--max_epoch', type=int, help='Maximum number of epochs', default=15)
parser.add_argument('-b', '--batch_size', type=int, help='Batch size', default=64)
parser.add_argument('-p', '--patience', type=int, help='The patience value for early stopping callback', default=15)
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


sources = pd.read_csv('Data/points_eau.csv', delimiter=';')
# sources = sources.head(7)

with open('performances.csv', 'a+', buffering=1) as perf_file:

    if not perf_file.tell():
        perf_file.write(',val_loss,test_loss,best_model_at_test,runtime_in_sec\n')
    else:
        perf = pd.read_csv('performances.csv', index_col=0)
        sources = sources[~sources.CODE_BSS.isin(perf.index)]

    for code_bss, bss_id in zip(sources.CODE_BSS, sources.BSS_ID):

        print('\n\nRunning on bss_id:', bss_id)

        start_time = time.time()

        # # Download data
        data = download_data(code_bss, f"Data/{bss_id}.csv")

        # # Preprocessing

        # ## Resampling the data to have a measure per day

        # start_date = '04-05-1970'
        # start_date = '09-08-1984'
        # start_date = '01-01-2000'
        start_date = data.index.min()

        data = data.loc[start_date:, :]

        new_index = pd.date_range(start_date, data.index[-1])

        missing_index = new_index.difference(data.index)

        print("\n\nThere are", missing_index.size, "missing measures over ", new_index.size, " => ", round(100 * missing_index.size/new_index.size, 2), "% of data")

        # Adding date features
        data = add_date_features(data)

        # Filling missing observations with Gaussian Processes

        imputer = build_imputer_model()

        imputer.fit(data[['year', 'month', 'quarter', 'weekday', 'day']], data['niveau_nappe_eau'])

        print('\n\nimputer.Score:', imputer.score(data[['year', 'month', 'quarter', 'weekday', 'day']], data['niveau_nappe_eau']), '\n\n')


        # Predict missing values
        missing_X = np.concatenate([missing_index.year.values.reshape((-1, 1)), 
                             missing_index.month.values.reshape((-1, 1)), 
                             missing_index.quarter.values.reshape((-1, 1)), 
                             missing_index.weekday.values.reshape((-1, 1)), 
                             missing_index.day.values.reshape((-1, 1))], axis=1)
        missing_X
        missing_y = imputer.predict(missing_X)

        cols = ['year', 'month', 'quarter', 'weekday', 'day', 'niveau_nappe_eau']
        missing_X_y = pd.DataFrame(data=np.concatenate([missing_X, np.expand_dims(missing_y, axis=1)], axis=1), 
                                   columns=cols, 
                                   index=missing_index)

        data_no_missing = data[cols].append(missing_X_y)
        data_no_missing = data_no_missing.sort_index()
        data_no_missing

        # ### Feature engineering
        # 
        # - Add temperature
        # - Add precipitation

        # First and second degree diff
        data_no_missing = add_derivate_features(data_no_missing, 'niveau_nappe_eau')

        # ### Split dataset 
        # 
        # $70\%$ for training, $20\%$ for validation and $10\%$ for test

        # In[14]:

        features = ['niveau_nappe_eau', 'year', 'month', 'quarter', 'weekday', 'day', 'diff_niveau_nappe_eau', 'diff30_niveau_nappe_eau', 'diff90_niveau_nappe_eau', 'diff180_niveau_nappe_eau', 'diff_diff_niveau_nappe_eau']

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
                                 train_df=train_df, val_df=val_df, test_df=test_df, label_columns=['niveau_nappe_eau'])

        no_suffled_window = WindowGenerator(input_width=input_width, 
                                 label_width=label_width, 
                                 shift=shift,
                                 batch_size=batch_size,
                                 train_df=train_df, val_df=val_df, test_df=test_df, label_columns=['niveau_nappe_eau'], shuffle=False)

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
                label_index = window_tmp.column_indices['niveau_nappe_eau'] if 'AR' in name else None

            models[name] = forecaster

        predictions = forecast(models[best_model], best_model, test_df, input_width, label_width, train_mean, train_std, label_index=label_index)
        
        predictions = pd.DataFrame({
            'CODE_BSS': code_bss,
            'DATE': predictions.index,
            'NIVEAU_PIEZO': predictions.niveau_nappe_eau * train_std.niveau_nappe_eau + train_mean.niveau_nappe_eau
        })

        if os.path.exists('submission.csv'):
            submission = pd.read_csv('submission.csv')
            submission = submission.append(predictions)
        else:
            submission = predictions

        submission.to_csv('submission.csv', index=False)

        end_time = time.time()

        perf_file.write(f"{code_bss},{best_val_score},{best_test_score},{best_model},{end_time - start_time}\n")

global_run_time = time.time() - global_start_time
global_run_time = round(global_run_time / 3600, 2)

print(f'All done. Took {global_run_time} H :)')

