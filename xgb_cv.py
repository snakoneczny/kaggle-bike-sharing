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

# Transform y
y_log = np.log(y + 1)

# Define targets
targets = [CASUAL, REGISTERED]
y_pred_all = {CASUAL: np.zeros(y.shape[0]), REGISTERED: np.zeros(y.shape[0])}

# CV
n_folds = 10
best_round = {CASUAL: np.zeros(n_folds), REGISTERED: np.zeros(n_folds)}
rmsle_fold = np.zeros(n_folds)
skf = KFold(y.shape[0], n_folds, shuffle=True, random_state=0)
i = 0
for train, test in skf:
    X_train, X_test = X.loc[train, :], X.loc[test, :]
    y_train, y_test = y.loc[train], y.loc[test]
    y_log_train, y_log_test = y_log.loc[train], y_log.loc[test]

    # Work with targets
    model = {}
    y_pred = np.zeros(X_test.shape[0])
    for target in targets:
        # XGBoost matrices
        # xg_train = xgb.DMatrix(X_train, label=y_train[target])
        # xg_test = xgb.DMatrix(X_test, label=y_test[target])
        xg_train = xgb.DMatrix(X_train, label=y_log_train[target])
        xg_test = xgb.DMatrix(X_test, label=y_log_test[target])

        # Train
        param = {'silent': 1, 'nthread': 8, 'objective': 'reg:linear',
                 'eta': 0.05, 'max_depth': 6, 'min_child_weight': 1, 'colsample_bytree': 1,
                 'subsample': 0.5, 'gamma': 0, 'alpha': 1, 'lambda': 1, 'lambda_bias': 0}
        n_rounds = 10000
        model[target] = xgb.train(param, xg_train, n_rounds, [(xg_train, 'train'), (xg_test, 'test')],
                                  #sle_obj, rmsle_evalerror,
                                  early_stopping_rounds=60)

        # Predict and clip values
        y_pred_target = model[target].predict(xg_test).clip(min=0)

        # Transform results
        y_pred_target = np.exp(y_pred_target) - 1

        # Add target predictions
        y_pred += y_pred_target

        # Save predictions and best iteration
        y_pred_all[target][test] = y_pred_target
        best_round[target][i] = model[target].best_iteration

    # Evaluate
    rmsle_fold[i] = rmsle(y_test[COUNT], y_pred)
    print 'Fold %d/%d, RMSLE = %f, best it. = %d, %d' % (
        i + 1, n_folds, rmsle_fold[i], best_round[CASUAL][i], best_round[REGISTERED][i])
    i += 1

# Show results
print best_round
print 'mean RMSLE = %f, ' % rmsle_fold.mean()
print 'mean best it. = %f, %f' % (best_round[CASUAL].mean(), best_round[REGISTERED].mean())

# Write cross validation results
write_results_extended(y_pred_all, 'cross-validation/xgb_%s.csv' % features, 'data/train.csv')
