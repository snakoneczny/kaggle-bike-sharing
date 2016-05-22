import sys

sys.path.append('../')
from utils import *
from sklearn.cross_validation import KFold
import xgboost as xgb

# Read data
train = pd.read_csv('data/train_simple.csv')

# Get X and y
X = train.drop(['casual', 'registered', 'count'], inplace=False, axis=1)
y = train['count']

# CV
n_folds = 10
rmsle_mean = 0.0
best_round = np.zeros(n_folds)
skf = KFold(y.shape[0], n_folds, shuffle=True, random_state=0)
i = 0
for train, test in skf:
    X_train, X_test = X.loc[train, :], X.loc[test, :]
    y_train, y_test = y.loc[train], y.loc[test]

    # XGBoost matrices
    xg_train = xgb.DMatrix(X_train, label=y_train)
    xg_test = xgb.DMatrix(X_test, label=y_test)
    param = {'objective': 'reg:linear', 'eval_metric': 'rmse', 'silent': 1, 'nthread': 4}
    n_rounds = 1000  # Just a big number to trigger early stopping and best iteration

    # Train
    bst = xgb.train(param, xg_train, n_rounds, [(xg_train, 'train'), (xg_test, 'test')], early_stopping_rounds=40)

    # Predict
    y_pred = bst.predict(xg_test)

    # Clip values lower than 0
    y_pred = y_pred.clip(min=0)

    # Evaluate
    rmsle_fold = rmsle(y_test, y_pred)
    rmsle_mean += rmsle_fold / n_folds
    best_round[i] = bst.best_iteration
    i += 1
    print 'Fold %d/%d, RMSLE = %f, best it. = %d' % (i, n_folds, rmsle_fold, bst.best_iteration)

# Show results
print best_round
print 'mean RMSLE = %f, mean best it. = %f' % (rmsle_mean, best_round.mean())
