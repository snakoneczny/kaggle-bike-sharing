from utils import *
from sklearn.cross_validation import KFold
from sklearn import preprocessing

np.random.seed(1227)

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, ActivityRegularization, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop
from keras.regularizers import l2, activity_l2, l1

# Parameters
features = NEURAL_NET

# Read data
train = pd.read_csv('data/train_%s.csv' % features)

# Get X and y
X = train.drop(['casual', 'registered', 'count',
                'month', 'day', 'season', 'weekday',  # 'season_ordered', 'hour',
                'weather', 'humidity_inv', 'windspeed_inv',
                ], inplace=False, axis=1)
y = train[['casual', 'registered', 'count']]

# CV
n_folds = 10
rmsle_fold = np.zeros(n_folds)
skf = KFold(y.shape[0], n_folds, shuffle=True, random_state=0)
i = 0
for train, test in skf:
    X_train, X_test = X.loc[train, :], X.loc[test, :]
    y_train, y_test = y.loc[train], y.loc[test]

    # Scale data
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define different targets
    # targets = ['count']
    targets = ['casual', 'registered']

    # Work with targets
    y_pred = np.zeros(X_test.shape[0])
    best_epoch = np.zeros(n_folds)
    for target in targets:
        y_train_target = y_train[target].as_matrix()
        y_test_target = y_test[target].as_matrix()

        # Define neural network
        model = Sequential()
        model.add(Dense(200, input_dim=X_train.shape[1]))
        model.add(Activation('relu'))
        model.add(Dropout(0.1))
        model.add(Dense(200))  # , W_regularizer=l2(0.1)))#, activity_regularizer=activity_l2(0.01)))
        model.add(Activation('relu'))
        model.add(Dropout(0.1))
        model.add(Dense(1))
        model.compile(loss='mean_squared_logarithmic_error', optimizer='rmsprop')

        # Train
        early_stopping = EarlyStopping(monitor='val_loss', patience=8, verbose=0)
        history = model.fit(X_train, y_train_target, validation_data=(X_test, y_test_target), shuffle=True,
                            callbacks=[early_stopping],
                            nb_epoch=160, batch_size=16)

        # Predict, reshape and clip values
        y_pred += model.predict(X_test).reshape(X_test.shape[0]).clip(min=0)

    # Evaluate
    rmsle_fold[i] = rmsle(y_test['count'], y_pred)
    print 'Fold %d/%d, RMSLE = %f' % (i + 1, n_folds, rmsle_fold[i])
    i += 1

# Show results
print 'RMSLE mean = %f, ' % rmsle_fold.mean()
