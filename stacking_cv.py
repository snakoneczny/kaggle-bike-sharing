from utils import *
from sklearn.cross_validation import KFold
from sklearn import preprocessing

np.random.seed(1227)

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, ActivityRegularization
from keras.callbacks import EarlyStopping
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
y = train[[CASUAL, REGISTERED, COUNT]]

# Define different targets
targets = [CASUAL, REGISTERED]
y_pred_all = {CASUAL: np.zeros(y.shape[0]), REGISTERED: np.zeros(y.shape[0])}

# Read predictions from previous models
models = ['rf_extended', 'xgb_extended', 'keras_neuralnet']
predictions_to_stack = pd.DataFrame()
for model in models:
    predictions_model = pd.read_csv('cross-validation/%s.csv' % model)
    for target in targets:
        predictions_to_stack[(model, target)] = predictions_model[target]

# CV
n_folds = 10
rmsle_fold = np.zeros(n_folds)
skf = KFold(y.shape[0], n_folds, shuffle=True, random_state=0)
i = 0
for train, test in skf:
    X_train, X_test = X.loc[train], X.loc[test]
    y_train, y_test = y.loc[train], y.loc[test]
    stacking_train, stacking_test = predictions_to_stack.loc[train], predictions_to_stack.loc[test]

    # Work with targets
    y_pred = np.zeros(X_test.shape[0])
    best_epoch = np.zeros(n_folds)
    for target in targets:
        X_train_target, X_test_target = X_train.copy(), X_test.copy()
        for model in models:
            X_train_target[(model, target)] = stacking_train[(model, target)]
            X_test_target[(model, target)] = stacking_test[(model, target)]

        # Scale data
        scaler = preprocessing.StandardScaler()
        X_train_target = scaler.fit_transform(X_train_target)
        X_test_target = scaler.transform(X_test_target)

        # Define neural network
        model = Sequential()
        model.add(Dense(200, input_dim=X_train_target.shape[1]))
        model.add(Activation('relu'))
        model.add(Dropout(0.1))
        model.add(Dense(200))  # , W_regularizer=l2(0.0001), activity_regularizer=activity_l2(0.01)))
        model.add(Activation('relu'))
        model.add(Dropout(0.1))
        model.add(Dense(1))
        model.compile(loss='mean_squared_logarithmic_error', optimizer='rmsprop')

        # Train
        early_stopping = EarlyStopping(monitor='val_loss', patience=8, verbose=0)
        history = model.fit(X_train_target, y_train[target], validation_data=(X_test_target, y_test[target]),
                            shuffle=True, callbacks=[early_stopping], nb_epoch=160, batch_size=16)

        # Predict, reshape and clip values
        y_pred_target = model.predict(X_test_target).reshape(X_test_target.shape[0]).clip(min=0)
        y_pred += y_pred_target

        # Save predictions
        y_pred_all[target][test] = y_pred_target

    # Evaluate
    rmsle_fold[i] = rmsle(y_test['count'], y_pred)
    print 'Fold %d/%d, RMSLE = %f' % (i + 1, n_folds, rmsle_fold[i])
    i += 1

# Show results
print 'RMSLE mean = %f, ' % rmsle_fold.mean()
