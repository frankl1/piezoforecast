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
import matplotlib.pyplot as plt

from gluonts.dataset.common import ListDataset
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx import Trainer
from gluonts.evaluation import make_evaluation_predictions
from gluonts.evaluation import Evaluator

print("load data")
rep_data = "../data_collection"
rep_results = "./"
datasetfile = 'dataset_nomissing_linear.csv'

data=pd.read_csv(rep_data+"/"+datasetfile, index_col=0)

prediction_length=93
freq='1D'

nb_series=len(data.bss.drop_duplicates())

print("prepare dataset")
# train dataset made of all the time series
train_ds = ListDataset(
    [{'target': data[data.bss==bss_id].p[:-prediction_length].to_numpy(), 'start': '2015-01-01'} for bss_id in data.bss.drop_duplicates()],
    freq=freq
)
# test dataset
test_ds = ListDataset(
    [{'target': data[data.bss==bss_id].p.to_numpy(), 'start': '2015-01-01'} for bss_id in data.bss.drop_duplicates()],
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
        epochs=20,
        learning_rate=1e-3,
        num_batches_per_epoch=32
    )
)

#learn the predictor
predictor = estimator.train(train_ds)


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
agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=nb_series)
item_metrics.item_id = data.bss.drop_duplicates().reset_index(drop=True)

print(item_metrics)
