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

def plot_prob_forecasts(ts_entry, forecast_entry):
    plot_length = 1000
    prediction_intervals = (50.0, 90.0)
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[-plot_length:].plot(ax=ax)  # plot the time series
    forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    plt.show()



rep_data = "../data_collection"
rep_results = "./"
datasetfile = 'dataset_nomissing_linear.csv'

data=pd.read_csv(rep_data+"/"+datasetfile, index_col=0)


prediction_length=93
freq='1D'


bss_id="00061X0117/PZ1"
dataset = data[data.bss==bss_id]

# train dataset
train_ds = ListDataset(
    [{'target': dataset.p[:-prediction_length].to_numpy(), 'start': '2015-01-01'}],
    freq=freq
)
# test dataset
test_ds = ListDataset(
    [{'target': dataset.p.to_numpy(), 'start': '2015-01-01'}],
    freq=freq
)

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

plot_prob_forecasts(tss[0],forecasts[0])

evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=1)
