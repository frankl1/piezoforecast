#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Thomas Guyet, Inria
@date: 2/2022
"""

import numpy as np
import pandas as pd
import time
import logging

from gluonts.dataset.common import ListDataset
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx import Trainer
from gluonts.evaluation import make_evaluation_predictions
from gluonts.evaluation import Evaluator

logging.getLogger("mxnet").addFilter(lambda record: False)

rep_data = "./data_collection"
rep_results = "./"
datasetfile = 'dataset_nomissing_linear.csv'

data=pd.read_csv(rep_data+"/"+datasetfile, index_col=0)
data = data[data.time<"2021-01-16"]

prediction_length=93
freq='1D'

metrics_list=[]
bss_idi=1
bss_nb=1329
for bss_id in data.bss.drop_duplicates():
    print( f"******* Process {bss_id} ({bss_idi}/{bss_nb}) *****" )
    bss_idi+=1
    dataset = data[data.bss==bss_id]
    
    for covariates in [['tp','e'],['tp'],['e'],[]]:
        # train dataset
        train_ds = ListDataset(
            [{
                    'target': dataset.p[:-prediction_length].to_numpy(), 
                    'start': '2015-01-01',
                    'feat_dynamic_real':dataset[covariates][:-prediction_length]
              }],
            freq=freq
        )
        # test dataset
        test_ds = ListDataset(
            [{
                    'target': dataset.p.to_numpy(), 
                    'start': '2015-01-01',
                    'feat_dynamic_real':dataset[covariates]
              }],
            freq=freq
        )
        """
        SimpleFeedForwardEstimator(
                num_hidden_dimensions=[10],
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
        """
                
        estimator = DeepAREstimator(
            prediction_length=prediction_length,
            context_length=100,
            freq=freq,
            trainer=Trainer(
                ctx="cpu",
                epochs=30,
                learning_rate=5e-4,
                num_batches_per_epoch=32
            )
        )
        start = time.time()
        #learn the predictor
        predictor = estimator.train(train_ds)
        end = time.time()
        learning_time = end - start
        
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
        agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=1)
        item_metrics.item_id=bss_id
        item_metrics['model']=estimator.__class__.__name__
        item_metrics['learningtime']=learning_time
        item_metrics['use_exo_rain']=('tp'in covariates)
        item_metrics['use_exo_evo']=('e'in covariates)
        item_metrics['rmse']=np.sqrt(item_metrics['MSE'])
        TN=np.mean((dataset.p[1:-prediction_length].to_numpy()-dataset.p[:-prediction_length-1].to_numpy())**2)
        item_metrics['rmsse']=np.sqrt(item_metrics['MSE']/TN)
        metrics_list.append(item_metrics)
    
        pd.concat(metrics_list).to_csv(estimator.__class__.__name__+"_local.csv.tmp")

metrics = pd.concat(metrics_list)
metrics.to_csv(estimator.__class__.__name__+"_local.csv")
