from utils import *
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestRegressor

# Parameters
features = EXTENDED

# Read data
train = pd.read_csv('data/train_%s.csv' % features)

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
rmsle_fold = np.zeros(n_folds)
skf = KFold(y.shape[0], n_folds, shuffle=True, random_state=0)
i = 0
for train, test in skf:
    X_train, X_test = X.loc[train, :], X.loc[test, :]
    y_train, y_test = y.loc[train], y.loc[test]
    y_log_train, y_log_test = y_log.loc[train], y_log.loc[test]

    # Work with targets
    y_pred = np.zeros(X_test.shape[0])
    for target in targets:
        # Train
        rf = RandomForestRegressor(random_state=0, n_jobs=8, n_estimators=400, max_features=None,
                                   max_depth=None, min_samples_split=1)
        rf.fit(X_train, y_log_train[target])

        # Predict and clip values
        y_pred_target = rf.predict(X_test).clip(min=0)

        # Transform results
        y_pred_target = np.exp(y_pred_target) - 1

        # Add target predictions
        y_pred += y_pred_target

        # Save predictions
        y_pred_all[target][test] = y_pred_target

    # Evaluate
    rmsle_fold[i] = rmsle(y_test[COUNT], y_pred)
    print 'Fold %d/%d, RMSLE = %f' % (i + 1, n_folds, rmsle_fold[i])
    i += 1

print 'mean RMSLE = %f' % rmsle_fold.mean()

# Write cross validation results
write_results_extended(y_pred_all, 'cross-validation/rf_%s.csv' % features, 'data/train.csv')
