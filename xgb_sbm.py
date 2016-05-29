from utils import *
import xgboost as xgb

# Parameters
features = EXTENDED

# Read data
train = pd.read_csv('data/train_%s.csv' % features)
test = pd.read_csv('data/test_%s.csv' % features)

# Get X and y
X_train = train.drop([COUNT, CASUAL, REGISTERED], inplace=False, axis=1)
y_train = train[[COUNT, CASUAL, REGISTERED]]
X_test = test

# Define targets
targets = [CASUAL, REGISTERED]
y_pred = {COUNT: np.zeros(X_test.shape[0]), CASUAL: np.zeros(X_test.shape[0]), REGISTERED: np.zeros(X_test.shape[0])}
n_rounds = {CASUAL: 840, REGISTERED: 720}

# Work with targets
model = {}
for target in targets:
    # XGBoost matrices
    xg_train = xgb.DMatrix(X_train.as_matrix(), label=y_train[target].as_matrix())
    xg_test = xgb.DMatrix(X_test.as_matrix())

    # Train
    param = {'silent': 1, 'nthread': 8, 'objective': 'reg:linear',
             'eta': 0.01, 'max_depth': 10, 'min_child_weight': 2, 'colsample_bytree': 1,
             'subsample': 0.5, 'gamma': 0, 'alpha': 8, 'lambda': 8, 'lambda_bias': 0}
    model[target] = xgb.train(param, xg_train, n_rounds[target], [(xg_train, 'train')], feval=rmsle_evalerror)

    # Predict and clip values lower than 0
    y_pred[target] = model[target].predict(xg_test).clip(min=0)
    y_pred[COUNT] += y_pred[target]

# Write submission
write_results(y_pred, 'submissions/xgb_%s.csv' % features, 'data/test.csv')
write_results_extended(y_pred, 'submissions/xgb_%s_stacking.csv' % features, 'data/test.csv')
