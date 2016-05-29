from utils import *
from sklearn import preprocessing

np.random.seed(1227)

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import xgboost as xgb

# Parameters
features = EXTENDED

# Read data
train = pd.read_csv('data/train_%s.csv' % features)
test = pd.read_csv('data/test_%s.csv' % features)

# Get X and y
# X_train = train.drop(['casual', 'registered', 'count',
#                       'month', 'day', 'season', 'weekday',
#                       'weather', 'humidity_inv', 'windspeed_inv',
#                       ], inplace=False, axis=1)
# X_test = test.drop(['month', 'day', 'season', 'weekday',
#                     'weather', 'humidity_inv', 'windspeed_inv',
#                     ], inplace=False, axis=1)
# y_train = train[[CASUAL, REGISTERED]]
X_train = train.drop([COUNT, CASUAL, REGISTERED], inplace=False, axis=1)
X_test = test
y_train = train[[COUNT, CASUAL, REGISTERED]]

# Define different targets
targets = [CASUAL, REGISTERED]
y_pred = {COUNT: np.zeros(X_test.shape[0]), CASUAL: np.zeros(X_test.shape[0]),
          REGISTERED: np.zeros(X_test.shape[0])}
xgb_n_rounds = {CASUAL: 480, REGISTERED: 500}

# Read predictions from previous models
models = ['rf_extended', 'xgb_extended', 'keras_neuralnet']
predictions_cv = pd.DataFrame()
predictions_sbm = pd.DataFrame()
for model in models:
    predictions_cv_model = pd.read_csv('cross-validation/%s.csv' % model)
    predictions_sbm_model = pd.read_csv('submissions/%s_stacking.csv' % model)
    for target in targets:
        predictions_cv[(model, target)] = predictions_cv_model[target]
        predictions_sbm[(model, target)] = predictions_sbm_model[target]

# Work with targets
for target in targets:
    X_train_target, X_test_target = X_train.copy(), X_test.copy()
    for model in models:
        X_train_target[(model, target)] = predictions_cv[(model, target)]
        X_test_target[(model, target)] = predictions_sbm[(model, target)]

    # Scale data
    scaler = preprocessing.StandardScaler()
    X_train_target = scaler.fit_transform(X_train_target)
    X_test_target = scaler.transform(X_test_target)

    # # Define neural network
    # model = Sequential()
    # model.add(Dense(200, input_dim=X_train_target.shape[1]))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.1))
    # model.add(Dense(200))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.1))
    # model.add(Dense(1))
    # model.compile(loss='mean_squared_logarithmic_error', optimizer='rmsprop')
    #
    # # Train
    # history = model.fit(X_train_target, y_train[target], shuffle=True, nb_epoch=50, batch_size=16)
    #
    # # Predict, reshape and clip values
    # y_pred[target] = model.predict(X_test_target).reshape(X_test_target.shape[0]).clip(min=0)
    # y_pred[COUNT] += y_pred[target]

    # XGBoost matrices
    xg_train = xgb.DMatrix(X_train_target.as_matrix(), label=y_train[target].as_matrix())
    xg_test = xgb.DMatrix(X_test_target.as_matrix())

    # Train
    param = {'silent': 1, 'nthread': 8, 'objective': 'reg:linear',
             'eta': 0.01, 'max_depth': 10, 'min_child_weight': 2, 'colsample_bytree': 1,
             'subsample': 0.5, 'gamma': 0, 'alpha': 2, 'lambda': 2, 'lambda_bias': 0}
    model = xgb.train(param, xg_train, xgb_n_rounds[target], [(xg_train, 'train')], feval=rmsle_evalerror)

    # Predict and clip values lower than 0
    y_pred[target] = model.predict(xg_test).clip(min=0)
    y_pred[COUNT] += y_pred[target]

# Write submission
write_submission(y_pred, 'submissions/stacking_xgb_%s.csv' % features)
