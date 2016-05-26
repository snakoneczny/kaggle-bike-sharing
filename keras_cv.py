from utils import *
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers.core import Dense, Activation


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
    targets = ['count']

    # Work with targets
    clf = {}
    y_pred = np.zeros(X_test.shape[0])
    for target in targets:

        model = Sequential()
        model.add(Dense(64, input_dim=X_train.shape[1]))
        # model.add(Activation('tanh'))
        model.add(Dense(1))
        # model.add(Activation('sigmoid'))
        model.compile(loss='mean_squared_logarithmic_error', optimizer='rmsprop')

        clf[target] = model

        # Train
        clf[target].fit(X_train.as_matrix(), y_train['count'].as_matrix(), nb_epoch=1, batch_size=32)

        # Evaluate
        score = model.evaluate(X_test.as_matrix(), y_test['count'].as_matrix(), batch_size=32)
        print score

        # Predict and clip values
        y_pred = clf[target].predict(X_test.as_matrix())
        y_pred = np.reshape(y_pred, y_pred.shape[0])

        print y_test['count']
        print y_pred
        print rmsle(y_test['count'], y_pred)
        exit(1)

    # Evaluate
    rmsle_fold = rmsle(y_test['count'], y_pred)
    rmsle_mean += rmsle_fold / n_folds
    i += 1
    print 'Fold %d/%d, RMSLE = %f' % (i, n_folds, rmsle_fold)

# Show results
print 'mean RMSLE = %f, ' % rmsle_mean
