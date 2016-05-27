import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt

SIMPLE = 'simple'
EXTENDED = 'extended'
NEURAL_NET = 'neuralnet'


def rmsle(y_true, y_pred):
    return sqrt(mean_squared_error(np.log(y_true + 1), np.log(y_pred + 1)))


def write_submission(y_pred, file_name):
    date = pd.read_csv('data/test.csv')['datetime']
    submission = pd.DataFrame(data={'datetime': date, 'count': y_pred})
    submission.to_csv(file_name, index=False)


def lse_obj(preds, dtrain):
    labels = dtrain.get_label()
    grad = 2 * np.divide(np.log(preds + 1) - np.log(labels + 1), preds + 1)
    hess = 2 * np.divide(1 - np.log(preds + 1) + np.log(labels + 1), np.power(preds + 1, 2))
    return grad, hess


def rmsle_evalerror(preds, dtrain):
    labels = dtrain.get_label()
    preds = preds.clip(min=0)
    return 'rmsle', sqrt(mean_squared_error(np.log(labels + 1), np.log(preds + 1)))
