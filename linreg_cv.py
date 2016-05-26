from utils import *
from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor, Lasso
from sklearn.preprocessing import PolynomialFeatures

# Read data
train = pd.read_csv('data/train_extended.csv')

# TODO: delete it: train = train[train.workingday == 1].reset_index(drop=True)

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

    # Transform polynomial base functions
    poly = PolynomialFeatures(degree=3, interaction_only=True, include_bias=True)
    X_train = poly.fit_transform(X_train)
    X_test = poly.transform(X_test)

    # Define different targets
    targets = ['casual', 'registered']

    # Train models
    clf = {}
    for target in targets:
        # clf[target] = Lasso(random_state=0, alpha=1.0, normalize=True, max_iter=1000, tol=0.0001, positive=False,
        #                     selection='cyclic')
        clf[target] = Ridge(random_state=0, alpha=1.0, normalize=True, max_iter=None, tol=0.001, solver='auto')
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
