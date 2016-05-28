from utils import *
import xgboost as xgb

# Parameters
features = EXTENDED

# Read data
train = pd.read_csv('data/train_%s.csv' % features)
test = pd.read_csv('data/test_%s.csv' % features)

# Get X and y
X_train = train.drop(['casual', 'registered', 'count'], inplace=False, axis=1)
y_train = train[['count', 'casual', 'registered']]
X_test = test

# Define different targets
targets = ['casual', 'registered']
n_rounds = {CASUAL: 480, REGISTERED: 660}

# Work with targets
clf = {}
y_pred = np.zeros(X_test.shape[0])
for target in targets:
    # XGBoost matrices
    xg_train = xgb.DMatrix(X_train.as_matrix(), label=y_train[target].as_matrix())
    xg_test = xgb.DMatrix(X_test.as_matrix())

    # Train
    param = {'silent': 1, 'nthread': 8, 'objective': 'reg:linear',
             'eta': 0.01, 'max_depth': 10, 'min_child_weight': 2, 'colsample_bytree': 1,
             'subsample': 0.5, 'gamma': 0, 'alpha': 2, 'lambda': 2, 'lambda_bias': 0}
    clf[target] = xgb.train(param, xg_train, n_rounds[target], [(xg_train, 'train')], feval=rmsle_evalerror)

    # Predict and clip
    y_pred += clf[target].predict(xg_test).clip(min=0)

# Write submission
write_submission(y_pred, 'submissions/xgb_%s.csv' % features)
