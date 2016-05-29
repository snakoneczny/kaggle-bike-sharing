from utils import *
from sklearn.cross_validation import KFold
import xgboost as xgb

# Parameters
features = EXTENDED

# Read data
train = pd.read_csv('data/train_extended.csv')

# Get X and y
X = train.drop([CASUAL, REGISTERED, COUNT], inplace=False, axis=1)
y = train[[CASUAL, REGISTERED, COUNT]]

# Define different targets
targets = [CASUAL, REGISTERED]
y_pred_all = {CASUAL: np.zeros(y.shape[0]), REGISTERED: np.zeros(y.shape[0])}

# CV
n_folds = 10
rmsle_mean = 0.0
best_round_casual = np.zeros(n_folds)
best_round_registered = np.zeros(n_folds)
skf = KFold(y.shape[0], n_folds, shuffle=True, random_state=0)
i = 0
for train, test in skf:
    X_train, X_test = X.loc[train, :], X.loc[test, :]
    y_train, y_test = y.loc[train], y.loc[test]

    # Work with targets
    clf = {}
    y_pred = np.zeros(X_test.shape[0])
    for target in targets:
        # XGBoost matrices
        xg_train = xgb.DMatrix(X_train, label=y_train[target])
        xg_test = xgb.DMatrix(X_test, label=y_test[target])

        # Train
        param = {'silent': 1, 'nthread': 8, 'objective': 'reg:linear',
                 'eta': 0.01, 'max_depth': 10, 'min_child_weight': 2, 'colsample_bytree': 1,
                 'subsample': 0.5, 'gamma': 0, 'alpha': 2, 'lambda': 2, 'lambda_bias': 0}
        n_rounds = 2000
        clf[target] = xgb.train(param, xg_train, n_rounds, [(xg_train, 'train'), (xg_test, 'test')],
                                feval=rmsle_evalerror, early_stopping_rounds=60)

        # Predict and clip values
        y_pred_target = clf[target].predict(xg_test).clip(min=0)
        y_pred += y_pred_target

        # Save predictions
        y_pred_all[target][test] = y_pred_target

    # Evaluate
    rmsle_fold = rmsle(y_test[COUNT], y_pred)
    rmsle_mean += rmsle_fold / n_folds
    best_round_casual[i] = clf[CASUAL].best_iteration
    best_round_registered[i] = clf[REGISTERED].best_iteration
    i += 1
    print 'Fold %d/%d, RMSLE = %f, best it. = %d, %d' % (
        i, n_folds, rmsle_fold, clf[CASUAL].best_iteration, clf[REGISTERED].best_iteration)

# Show results
print best_round_casual, best_round_registered
print 'mean RMSLE = %f, ' % rmsle_mean
print 'mean best it. = %f, %f' % (best_round_casual.mean(), best_round_registered.mean())

# Write cross validation results
date = pd.read_csv('data/train.csv')['datetime']
cv_results = pd.DataFrame(
    data={'datetime': date, CASUAL: y_pred_all[CASUAL], REGISTERED: y_pred_all[REGISTERED]})
cv_results.to_csv('cross-validation/xgb_%s.csv' % features, index=False)
