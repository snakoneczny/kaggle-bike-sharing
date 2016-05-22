import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt


SIMPLE = 'simple'
CIRCULAR = 'circular'


def rmsle(y_true, y_pred):
    return sqrt(mean_squared_error(np.log(y_true + 1), np.log(y_pred + 1)))


def write_submission(y_pred, file_name):
    date = pd.read_csv('data/test.csv')['datetime']
    submission = pd.DataFrame(data={'datetime': date, 'count': y_pred})
    submission.to_csv(file_name, index=False)