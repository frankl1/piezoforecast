#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Thomas Guyet, Inria
@date: 2/2022
"""

import numpy as np
import pandas as pd
import time
import os
from pathlib import Path

from gluonts.dataset.common import ListDataset
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx import Trainer
from gluonts.evaluation import make_evaluation_predictions
from gluonts.evaluation import Evaluator

print("load data")
rep_data = "../data_collection"
rep_results = "./"
rep_models = "./models"
datasetfile = 'dataset_nomissing_linear.csv'

try:
    os.mkdir(rep_models)
except:
    None

data=pd.read_csv(rep_data+"/"+datasetfile, index_col=0)
data = data[data.time<"2021-01-16"]

# stations pour récupérer le code LISA et avoir les données de la BDLisa
stations=pd.read_csv(rep_data+"/stations.csv", index_col=0)
stations = pd.merge(stations, data.bss.drop_duplicates(), on="bss", how="right")[['bss','EtatEH', 'NatureEH', 'MilieuEH','ThemeEH', 'OrigineEH']]
# replace NaN to keep the exact same number of time series (1195 remining otherwise)
stations.replace({np.NaN:0}, inplace=True)
stations.replace({'X':0}, inplace=True)
stations.set_index('bss', inplace=True)

#stations.dropna(inplace=True) 

prediction_length=93
freq='1D'

nb_series=len(data.bss.drop_duplicates())

print('Pre-compute TN')
TN=[{'bss':bss_id,'TN':np.mean((data[data.bss==bss_id].iloc[1:-prediction_length].p.to_numpy()-data[data.bss==bss_id].iloc[:-prediction_length-1].p.to_numpy())**2)} for bss_id in data.bss.drop_duplicates()]
TN=pd.DataFrame(TN)

bdlisa=False

metrics_list=[]
#for covariates in [[] ,['tp'],['e'], ['tp','e']]:
#for covariates in [[],['tp'],['e'], ['tp','e']]:
for covariates in [[]]:
    print("prepare dataset with covariates: "+ ', '.join(covariates) +".")
    if bdlisa:
        # train dataset made of all the time series
        train_ds = ListDataset(
            [{'target': data[data.bss==bss_id].p[:-prediction_length].to_numpy(), 
              'start': '2015-01-01', 
              'feat_dynamic_real':data[data.bss==bss_id][covariates][:-prediction_length],
              'feat_static_cat': stations.loc[bss_id],
              } for bss_id in data.bss.drop_duplicates()],
            freq=freq
        )
        # test dataset
        test_ds = ListDataset(
            [{'target': data[data.bss==bss_id].p.to_numpy(), 
              'start': '2015-01-01',
              'feat_dynamic_real':data[data.bss==bss_id][covariates],
              'feat_static_cat': stations.loc[bss_id]
              } for bss_id in data.bss.drop_duplicates()],
            freq=freq
        )
        estimator_specname="_global_bdlisa_"+"_".join(covariates)
    else:
        # train dataset made of all the time series
        train_ds = ListDataset(
            [{'target': data[data.bss==bss_id].p[:-prediction_length].to_numpy(), 
              'start': '2015-01-01', 
              'feat_dynamic_real':data[data.bss==bss_id][covariates][:-prediction_length]
              } for bss_id in data.bss.drop_duplicates()],
            freq=freq
        )
        # test dataset
        test_ds = ListDataset(
            [{'target': data[data.bss==bss_id].p.to_numpy(), 
              'start': '2015-01-01',
              'feat_dynamic_real':data[data.bss==bss_id][covariates]
              } for bss_id in data.bss.drop_duplicates()],
            freq=freq
        )
        estimator_specname="_global_"+"_".join(covariates)
    
        
    
    print("Learn global model")
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
    
    #save model
    estimator_name=estimator.__class__.__name__+estimator_specname
    try:
        os.mkdir(os.path.join(rep_models,estimator_name))
    except:
        None
    predictor.serialize(Path(os.path.join(rep_models,estimator_name)))
    
    print("Forecast")
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
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=nb_series)
    item_metrics.item_id = data.bss.drop_duplicates().reset_index(drop=True)
    
    item_metrics['model']=estimator.__class__.__name__
    item_metrics['learningtime']=learning_time
    item_metrics['use_exo_rain']=('tp'in covariates)
    item_metrics['use_exo_evo']=('e'in covariates)
    item_metrics['use_exo_lisa']=bdlisa
    item_metrics['rmse']=np.sqrt(item_metrics['MSE'])
    item_metrics=pd.merge(item_metrics, TN, left_on="item_id", right_on="bss").drop("bss",axis=1)
    item_metrics['rmsse']=np.sqrt(item_metrics['MSE']/item_metrics['TN'])
    metrics_list.append(item_metrics)

    pd.concat(metrics_list).to_csv(estimator.__class__.__name__+"_global.csv.tmp")

metrics = pd.concat(metrics_list)
metrics.to_csv(estimator.__class__.__name__+"_global.csv")
