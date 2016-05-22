import sys

sys.path.append('../')
from utils import *
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestRegressor

# Read data
train = pd.read_csv('data/train_simple.csv')

# Get X and y
X = train.drop(['casual', 'registered', 'count'], inplace=False, axis=1)
y = train[['casual', 'registered', 'count']]

# CV
n_folds = 10
rmsle_mean = 0.0
skf = KFold(y.shape[0], n_folds, shuffle=True, random_state=0)
i = 0
for train, test in skf:
    X_train, X_test = X.loc[train, :], X.loc[test, :]
    y_train, y_test = y.loc[train], y.loc[test]

    # Define different targets
    targets = ['casual', 'registered']

    # Train
    clf = {}
    for target in targets:
        clf[target] = RandomForestRegressor(random_state=0, n_jobs=16, n_estimators=100, max_features=None,
                                            max_depth=None, min_samples_split=1)
        clf[target].fit(X_train, y_train[target])

    # Predict
    y_pred = clf['casual'].predict(X_test) + clf['registered'].predict(X_test)

    # Clip values lower than 0
    y_pred = y_pred.clip(min=0)

    # Evaluate
    rmsle_fold = rmsle(y_test['count'], y_pred)
    rmsle_mean += rmsle_fold / n_folds
    i += 1
    print 'Fold %d/%d, RMSLE = %f' % (i, n_folds, rmsle_fold)  # Show results

print 'mean RMSLE = %f' % rmsle_mean
