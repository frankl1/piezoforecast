#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Thomas Guyet, Inria
@date: 2/2022
"""

import mxnet as mx
from mxnet import gluon
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

from gluonts.dataset.common import ListDataset
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx import Trainer
from gluonts.evaluation import make_evaluation_predictions
from gluonts.evaluation import Evaluator

print("load data")
rep_data = "data_collection"
rep_results = "./"
datasetfile = 'dataset_nomissing_linear.csv'

data=pd.read_csv(rep_data+"/"+datasetfile, index_col=0)
data = data[data.time<"2021-01-16"]

prediction_length=93
freq='1D'

nb_series=len(data.bss.drop_duplicates())

print('Pre-compute TN')
TN=[{'bss':bss_id,'TN':np.mean((data[data.bss==bss_id].iloc[1:-prediction_length].p.to_numpy()-data[data.bss==bss_id].iloc[:-prediction_length-1].p.to_numpy())**2)} for bss_id in data.bss.drop_duplicates()]
TN=pd.DataFrame(TN)

metrics_list=[]

indices = np.arange(nb_series)

kf = KFold(n_splits=4, shuffle=True)

for fold, (train_indices, test_indices) in enumerate(kf.split(indices)):
    #for covariates in [[] ,['tp'],['e'], ['tp','e']]:
    for covariates in [[]]:
        print("prepare dataset with covariates: "+ ', '.join(covariates) +".")
        # train dataset made of all the time series
        train_ds = ListDataset(
            [{'target': data[data.bss==bss_id].p[:-prediction_length].to_numpy(), 
            'start': '2015-01-01', 
            'feat_dynamic_real':data[data.bss==bss_id][covariates][:-prediction_length]} for bss_id in data.bss.drop_duplicates().iloc[train_indices]],
            freq=freq
        )
        # test dataset
        test_ds = ListDataset(
            [{'target': data[data.bss==bss_id].p.to_numpy(), 
            'start': '2015-01-01',
            'feat_dynamic_real':data[data.bss==bss_id][covariates]} for bss_id in data.bss.drop_duplicates().iloc[test_indices]],
            freq=freq
        )
        
        
        print("learn model")
        """
        estimator = SimpleFeedForwardEstimator(
            num_hidden_dimensions=[10],
            prediction_length=prediction_length,
            context_length=100,
            freq=freq,
            trainer=Trainer(
                ctx="cpu",
                epochs=150,
                learning_rate=1e-3,
                num_batches_per_epoch=32
            )
        )
        """
        estimator = DeepAREstimator(
            prediction_length=prediction_length,
            context_length=100,
            freq=freq,
            trainer=Trainer(
                ctx="cpu",
                epochs=30,
                learning_rate=1e-3,
                num_batches_per_epoch=32
            )
        )
        
        #learn the predictor
        start = time.time()
        predictor = estimator.train(train_ds)
        end = time.time()
        learning_time = end - start
        
        
        print("forecast")
        #do forecast
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=test_ds,  # test dataset
            predictor=predictor,  # predictor
            num_samples=100,  # number of sample paths we want for evaluation
        )
        
        # evaluate the results
        forecasts = list(forecast_it)
        tss = list(ts_it)
        
        evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
        agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(test_indices))
        item_metrics.item_id = data.bss.drop_duplicates().reset_index(drop=True)
        
        item_metrics['model']=estimator.__class__.__name__
        item_metrics['learningtime']=learning_time
        item_metrics['use_exo_rain']=('tp'in covariates)
        item_metrics['use_exo_evo']=('e'in covariates)
        item_metrics['rmse']=np.sqrt(item_metrics['MSE'])
        item_metrics=pd.merge(item_metrics, TN, left_on="item_id", right_on="bss").drop("bss",axis=1)
        item_metrics['rmsse']=np.sqrt(item_metrics['MSE']/item_metrics['TN'])
        item_metrics['fold']=fold
        metrics_list.append(item_metrics)

        fname = estimator.__class__.__name__+f"_global_cv.csv"

        metrics = None
        if os.path.isfile(fname):
            metrics = pd.read_csv(fname)
            metrics = metrics.append(pd.DataFrame(item_metrics))
        else:
            metrics = pd.DataFrame(item_metrics)
        metrics.to_csv(fname)
