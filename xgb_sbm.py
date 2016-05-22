import sys

sys.path.append('../')
from utils import *
import xgboost as xgb

# Parameters
engineering_type = SIMPLE

# Read data
train = pd.read_csv('data/train_%s.csv' % engineering_type)
test = pd.read_csv('data/test_%s.csv' % engineering_type)

# Get X and y
X_train = train.drop(['casual', 'registered', 'count'], inplace=False, axis=1)
y_train = train['count']
X_test = test

# XGBoost matrices
xg_train = xgb.DMatrix(X_train, label=y_train)
xg_test = xgb.DMatrix(X_test)

# Train
param = {'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': 1, 'nthread': 4}
n_rounds = 190
bst = xgb.train(param, xg_train, n_rounds, [(xg_train, 'train')])

# Predict
y_pred = bst.predict(xg_test)

# Clip values lower than 0
y_pred = y_pred.clip(min=0)

# Write submission
write_submission(y_pred, 'xgb_%s.csv' % engineering_type)
