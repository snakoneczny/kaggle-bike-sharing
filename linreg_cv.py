from utils import *
from sklearn.cross_validation import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

# Read data
train = pd.read_csv('data/train_extended.csv')

# Get X and y
X = train.drop([CASUAL, REGISTERED, COUNT], inplace=False, axis=1)
y = train[[CASUAL, REGISTERED, COUNT]]

# Define targets
targets = [CASUAL, REGISTERED]

# CV
n_folds = 10
rmsle_fold = np.zeros(n_folds)
skf = KFold(y.shape[0], n_folds, shuffle=True, random_state=0)
i = 0
for train, test in skf:
    X_train, X_test = X.loc[train, :], X.loc[test, :]
    y_train, y_test = y.loc[train], y.loc[test]

    # Transform polynomial base functions
    poly = PolynomialFeatures(degree=3, interaction_only=True, include_bias=True)
    X_train = poly.fit_transform(X_train)
    X_test = poly.transform(X_test)

    # Train models
    model = {}
    for target in targets:
        # clf[target] = Lasso(random_state=0, alpha=1.0, normalize=True, max_iter=1000, tol=0.0001, positive=False,
        #                     selection='cyclic')
        model[target] = Ridge(random_state=0, alpha=1.0, normalize=True, max_iter=None, tol=0.001, solver='auto')
        model[target].fit(X_train, y_train[target])

    # Predict and clip
    y_pred = model[CASUAL].predict(X_test).clip(min=0) + model[REGISTERED].predict(X_test).clip(min=0)

    # Evaluate
    rmsle_fold[i] = rmsle(y_test[COUNT], y_pred)
    print 'Fold %d/%d, RMSLE = %f' % (i + 1, n_folds, rmsle_fold[i])
    i += 1

# Show results
print 'mean RMSLE = %f' % rmsle_fold.mean()
