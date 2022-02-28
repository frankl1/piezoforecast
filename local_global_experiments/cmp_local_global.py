#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Thomas Guyet, Inria
@date: 2/2022
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path

from gluonts.dataset.common import ListDataset
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx import Trainer
from gluonts.evaluation import make_evaluation_predictions
from gluonts.evaluation import Evaluator

import matplotlib.pyplot as plt

from itertools import islice
def plot_forecasts(tss, forecasts, past_length, num_plots):
    for target, forecast in islice(zip(tss, forecasts), num_plots):
        ax = target[-past_length:].plot(figsize=(12, 5), linewidth=2)
        forecast.plot(color='g')
        plt.grid(which='both')
        plt.legend(["observations", "median prediction", "90% confidence interval", "50% confidence interval"])
        plt.show()
        
rep_data = "../data_collection"
rep_results = "./"
datasetfile = 'dataset_nomissing_linear.csv'

data=pd.read_csv(rep_data+"/"+datasetfile, index_col=0)
data = data[data.time<"2021-01-16"]

prediction_length=93
freq='1D'

#bss_id = '00068X0010/F295'
bss_id=list_bss[1]
covariates = []

dataset = data[data.bss==bss_id]


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
#learn the predictor
predictor = estimator.train(train_ds)

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
item_metrics['rmse']=np.sqrt(item_metrics['MSE'])
TN=np.sum((dataset.p[1:-prediction_length].to_numpy()-dataset.p[:-prediction_length-1].to_numpy())**2)
item_metrics['rmsse']=np.sqrt(item_metrics['MSE']/TN)


plot_forecasts(tss, forecasts, past_length=150, num_plots=1)