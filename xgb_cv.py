from utils import *
from sklearn.cross_validation import KFold
import xgboost as xgb

# Read data
train = pd.read_csv('data/train_extended.csv')

# Get X and y
X = train.drop(['casual', 'registered', 'count'], inplace=False, axis=1)
y = train[['casual', 'registered', 'count']]

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

    # Define different targets
    targets = ['casual', 'registered']

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
        y_pred += clf[target].predict(xg_test).clip(min=0)

    # Evaluate
    rmsle_fold = rmsle(y_test['count'], y_pred)
    rmsle_mean += rmsle_fold / n_folds
    best_round_casual[i] = clf['casual'].best_iteration
    best_round_registered[i] = clf['registered'].best_iteration
    i += 1
    print 'Fold %d/%d, RMSLE = %f, best it. = %d, %d' % (
        i, n_folds, rmsle_fold, clf['registered'].best_iteration, clf['casual'].best_iteration)

# Show results
print best_round_casual, best_round_registered
print 'mean RMSLE = %f, ' % rmsle_mean
print 'mean best it. = %f, %f' % (best_round_casual.mean(), best_round_registered.mean())
