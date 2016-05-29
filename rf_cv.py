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

# Define different targets
targets = [CASUAL, REGISTERED]
y_pred_all = {CASUAL: np.zeros(y.shape[0]), REGISTERED: np.zeros(y.shape[0])}

# CV
rmsle_mean = 0.0
n_folds = 10
skf = KFold(y.shape[0], n_folds, shuffle=True, random_state=0)
i = 0
for train, test in skf:
    X_train, X_test = X.loc[train, :], X.loc[test, :]
    y_train, y_test = y.loc[train], y.loc[test]

    # Work with targets
    y_pred = np.zeros(X_test.shape[0])
    for target in targets:
        # Train
        rf = RandomForestRegressor(random_state=0, n_jobs=8, n_estimators=100, max_features=None,
                                   max_depth=None, min_samples_split=1)
        rf.fit(X_train, y_train[target])

        # Predict and clip values
        y_pred_target = rf.predict(X_test).clip(min=0)
        y_pred += y_pred_target

        # Save predictions
        y_pred_all[target][test] = y_pred_target

    # Evaluate
    rmsle_fold = rmsle(y_test[COUNT], y_pred)
    rmsle_mean += rmsle_fold / n_folds
    i += 1
    print 'Fold %d/%d, RMSLE = %f' % (i, n_folds, rmsle_fold)  # Show results

print 'mean RMSLE = %f' % rmsle_mean

# Write cross validation results
date = pd.read_csv('data/train.csv')['datetime']
cv_results = pd.DataFrame(
    data={'datetime': date, CASUAL: y_pred_all[CASUAL], REGISTERED: y_pred_all[REGISTERED]})
cv_results.to_csv('cross-validation/rf_%s.csv' % features, index=False)
