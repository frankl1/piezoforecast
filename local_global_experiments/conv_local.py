#!/usr/bin/env python
# coding: utf-8

# In[1]:


import code
from xml.sax.handler import all_features
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
parser.add_argument('dataset', type=str, help='the dataset file')
parser.add_argument('out_file', type=str, help="the result file")

args = parser.parse_args()

# dataset
dataset = args.dataset

# output file
out_file = args.out_file

# The horizon size
label_width = 93

# history_factor
h_factor = 1.25

# An history length of 
# input_width = int(label_width * h_factor)
input_width = 100

# Number of feature values to predict at once
shift = label_width

# batch size
batch_size = args.batch_size

MAX_EPOCHS = args.max_epoch

patience = args.patience

verbose = args.verbose

#sources = pd.read_csv('../data_collection/dataset_stations.csv', delimiter=',', index_col=0)

all_data = pd.read_csv(dataset, delimiter=',', index_col=0, parse_dates=['time'])
all_data.set_index('bss', drop=True, inplace=True)
all_data = all_data[all_data.time<"2021-01-16"]

sources = all_data.index.unique()

columns = ['bss_code','model','rmse_train','rmse_test','rmsse_train','rmsse_test','learningtime', 'use_exo_rain', 'use_exo_evo'] + [f'h{i}' for i in range(1, 94)]

to_skip = []
if os.path.isfile(out_file):
    to_skip = pd.read_csv(out_file, usecols=['model', 'bss_code'])
    to_skip = (to_skip.bss_code + to_skip.model).values

with open(out_file, 'a+', buffering=1) as f:

    if os.stat(out_file).st_size == 0:
        f.write(','.join(columns) + '\n')

    count = 0

    model = 'Conv'

    for code_bss in sources:
        count += 1

        # # Retrieve data
        if code_bss+model in to_skip:
            continue

        print('\n\nRunning on piezo No:', count, 'ID:', code_bss)
            
        data = all_data.loc[code_bss].set_index('time')
        data = data.sort_index()
        
        # select depth col
        data_no_missing = data.copy()
        data_no_missing.e = data_no_missing.e.interpolate(method='linear') # evapotranspiration
        data_no_missing.p = data_no_missing.tp.interpolate(method='linear') # pricipitation

        # Adding date features
        data_no_missing = add_date_features(data)

        # First and second degree diff
        data_no_missing = add_derivate_features(data_no_missing, 'p')
        data_no_missing = add_derivate_features(data_no_missing, 'e')
        data_no_missing = add_derivate_features(data_no_missing, 'tp')

        # ### Split dataset 
        # 
        # $70\%$ for training, $20\%$ for validation and $10\%$ for test

        # In[14]:

        base_features = ['p', 'year', 'month', 'quarter', 'weekday', 'day', 'diff_p', 'diff30_p', 'diff90_p', 'diff180_p', 'diff_diff_p']
        
        lines = ''

        scaled_factor = np.mean(data_no_missing.p.diff().dropna()**2) # RMSSE factor

        for cov_list in [[], ['e'], ['tp'], ['e', 'tp']]:
            all_features = base_features.copy()
            
            for cov in cov_list:
                all_features += [cov, 'diff_'+cov, 'diff30_'+cov, 'diff90_'+cov, 'diff180_'+cov, 'diff_diff_'+cov]
                
            df = data_no_missing[all_features]

            n_features = len(all_features)

            n = df.shape[0]

            n_train = n - label_width - input_width # keep one sample for test
            len_train = int(n_train * 0.8) # 80% for training and 20% for validation
            #len_val = int(n_train * 0.2)

            train_df = df[:len_train]
            val_df = df[len_train: n_train]
            
            test_df = df[-(label_width+input_width):]

            print('\tTrain shape:', train_df.shape)
            print('\Val shape:', val_df.shape)
            print('\tTest shape:', test_df.shape)
            print('\tNumber of observations:', n) 
            print('\tFeatures:', all_features, 'number:', n_features)

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

            # ## Model 

            conv_model = tf.keras.Sequential([
                tf.keras.layers.Conv1D(filters=256, kernel_size=7, activation='relu', padding='same'),
                tf.keras.layers.Conv1D(filters=128, kernel_size=7, activation='relu', padding='same'),
                tf.keras.layers.Conv1D(filters=64, kernel_size=7, activation='relu', padding='same'),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1)),
                tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1)),
                tf.keras.layers.Dense(units=label_width),
                tf.keras.layers.Reshape((label_width, -1))
            ], name='conv_model')   

            try:
                start_time = time.time()
                history = compile_and_fit(conv_model, window, patience=patience, epochs=MAX_EPOCHS, verbose=verbose)
                learning_time = time.time() - start_time

                mse_train = conv_model.evaluate(window.val, verbose=verbose)
                mse_test = conv_model.evaluate(window.test, verbose=verbose)

                msse_train = mse_train / scaled_factor
                msse_test = mse_test / scaled_factor

                predictions = conv_model.predict(window.test, verbose=verbose).squeeze() * train_std.p + train_mean.p

                lines += ','.join(np.array([code_bss, model, np.sqrt(mse_train), np.sqrt(mse_test), np.sqrt(msse_train), np.sqrt(msse_test), learning_time, 'tp' in cov_list, 'e' in cov_list] + predictions.tolist(), dtype=str))+'\n'
            except Exception as e:
                print("Exception on piezo ", code_bss, "with covariates:", cov_list, "Error:", e)
                lines += f"{code_bss},{model},,,,,,{'tp' in cov_list},{'e' in cov_list}" + (','*label_width)+'\n'
        f.write(lines)
