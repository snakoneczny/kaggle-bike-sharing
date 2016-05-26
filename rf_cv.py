from utils import *
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestRegressor

# Parameters
features = EXTENDED

# Read data
train = pd.read_csv('data/train_%s.csv' % features)

# Get X and y
X = train.drop(['casual', 'registered', 'count'], inplace=False, axis=1)
y = train[['casual', 'registered', 'count']]

# CV
rmsle_mean = 0.0
n_folds = 10
skf = KFold(y.shape[0], n_folds, shuffle=True, random_state=0)
i = 0
for train, test in skf:
    X_train, X_test = X.loc[train, :], X.loc[test, :]
    y_train, y_test = y.loc[train], y.loc[test]

    # Work on subproblems
    # y_test = y_test[X_test.workingday == 0].reset_index(drop=True).append(
    #     y_test[X_test.workingday == 1].reset_index(drop=True), ignore_index=True)
    # y_pred = []
    # for workingday in xrange(2):
    #     X_train_sub = X_train[X_train.workingday == workingday].reset_index(drop=True)
    #     y_train_sub = y_train[X_train.workingday == workingday].reset_index(drop=True)
    #     X_test_sub = X_test[X_test.workingday == workingday].reset_index(drop=True)

    # Define different targets
    targets = ['casual', 'registered']

    # Train
    clf = {}
    for target in targets:
        clf[target] = RandomForestRegressor(random_state=0, n_jobs=8, n_estimators=100, max_features=None,
                                            max_depth=None, min_samples_split=1)
        clf[target].fit(X_train, y_train[target])

    # Predict
    y_pred = clf['casual'].predict(X_test) + clf['registered'].predict(X_test)

    # Clip values lower than 0
    y_pred = y_pred.clip(min=0)

    # Store sub predictions
    # y_pred = np.concatenate((y_pred, y_pred_sub))

    # Evaluate
    rmsle_fold = rmsle(y_test['count'], y_pred)
    rmsle_mean += rmsle_fold / n_folds
    i += 1
    print 'Fold %d/%d, RMSLE = %f' % (i, n_folds, rmsle_fold)  # Show results

print 'mean RMSLE = %f' % rmsle_mean