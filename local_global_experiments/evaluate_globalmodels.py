#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 09:58:06 2022

@author: tguyet
"""

from gluonts.model.predictor import Predictor

import numpy as np
import pandas as pd
import os
from pathlib import Path

from gluonts.dataset.common import ListDataset
from gluonts.evaluation import make_evaluation_predictions
from gluonts.evaluation import Evaluator

import matplotlib.pyplot as plt

print("Load model")
rep_models = "./models"
model="DeepAREstimator_global_bdlisa_"
predictor = Predictor.deserialize(Path(os.path.join(rep_models,model)))
covariates=[]
bdlisa=True
#TODO: save these parameters in a pkl while saving the model

print("load data")
rep_data = "../data_collection"
rep_results = "./"
datasetfile = 'dataset_nomissing_linear.csv'

data=pd.read_csv(rep_data+"/"+datasetfile, index_col=0)
data = data[data.time<"2021-01-16"]

# stations pour récupérer le code LISA et avoir les données de la BDLisa
stations=pd.read_csv(rep_data+"/stations.csv", index_col=0)
stations = pd.merge(stations, data.bss.drop_duplicates(), on="bss", how="right")[['bss','EtatEH', 'NatureEH', 'MilieuEH','ThemeEH', 'OrigineEH']]
# replace NaN to keep the exact same number of time series (1195 remining otherwise)
stations.replace({np.NaN:0}, inplace=True)
stations.replace({'X':0}, inplace=True)
stations.set_index("bss",inplace=True)

prediction_length=93
freq='1D'

list_bss=['00068X0010/F295','11064X0013/ALISO','00103X0322/F']

if bdlisa:
    test_ds = ListDataset(
        [{'target': data[data.bss==bss_id].p.to_numpy(), 
          'start': '2015-01-01',
          'feat_dynamic_real':data[data.bss==bss_id][covariates],
          'feat_static_cat': stations.loc[bss_id]
          } for bss_id in list_bss],
        freq=freq
    )
else:
    test_ds = ListDataset(
        [{'target': data[data.bss==bss_id].p.to_numpy(), 
          'start': '2015-01-01',
          'feat_dynamic_real':data[data.bss==bss_id][covariates],
          'feat_static_cat': stations.loc[bss_id]
          } for bss_id in list_bss],
        freq=freq
    )

forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds,  # test dataset
    predictor=predictor,  # predictor
    num_samples=100,  # number of sample paths we want for evaluation
)

# evaluate the results
forecasts = list(forecast_it)
tss = list(ts_it)

evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(list_bss))
item_metrics.item_id = list_bss
item_metrics['rmse']=np.sqrt(item_metrics['MSE'])
TN=[{'bss':bss_id,'TN':np.sum((data[data.bss==bss_id].iloc[1:-prediction_length].p.to_numpy()-data[data.bss==bss_id].iloc[:-prediction_length-1].p.to_numpy())**2)} for bss_id in list_bss]
TN=pd.DataFrame(TN)
item_metrics=pd.merge(item_metrics, TN, left_on="item_id", right_on="bss").drop("bss",axis=1)
item_metrics['rmsse']=np.sqrt(item_metrics['MSE']/item_metrics['TN'])


from itertools import islice
def plot_forecasts(tss, forecasts, past_length, num_plots):
    for target, forecast in islice(zip(tss, forecasts), num_plots):
        ax = target[-past_length:].plot(figsize=(12, 5), linewidth=2)
        forecast.plot(color='g')
        plt.grid(which='both')
        plt.legend(["observations", "median prediction", "90% confidence interval", "50% confidence interval"])
        plt.show()

plot_forecasts(tss, forecasts, past_length=150, num_plots=len(list_bss))


