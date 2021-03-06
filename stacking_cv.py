from utils import *
from sklearn.cross_validation import KFold
import xgboost as xgb
# np.random.seed(1227)
#
# from sklearn import preprocessing
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Dropout
# from keras.callbacks import EarlyStopping

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

# Read predictions from previous models
models_stacking = ['rf_extended', 'xgb_extended', 'keras_neuralnet']
predictions = pd.DataFrame()
for model_stacking in models_stacking:
    predictions_model = pd.read_csv('cross-validation/%s.csv' % model_stacking)
    for target in targets:
        predictions[(model_stacking, target)] = predictions_model[target]

# CV
n_folds = 10
rmsle_fold = np.zeros(n_folds)
best_round = {CASUAL: np.zeros(n_folds), REGISTERED: np.zeros(n_folds)}
skf = KFold(y.shape[0], n_folds, shuffle=True, random_state=0)
i = 0
for train, test in skf:
    X_train, X_test = X.loc[train], X.loc[test]
    y_train, y_test = y.loc[train], y.loc[test]
    y_log_train, y_log_test = y_log.loc[train], y_log.loc[test]
    predictions_train, predictions_test = predictions.loc[train], predictions.loc[test]

    # Work with targets
    model = {}
    y_pred = np.zeros(X_test.shape[0])
    for target in targets:
        # X_train_target, X_test_target = X_train.copy(), X_test.copy()
        X_train_target, X_test_target = pd.DataFrame(), pd.DataFrame()
        for model_stacking in models_stacking:
            X_train_target[(model_stacking, target)] = predictions_train[(model_stacking, target)]
            X_test_target[(model_stacking, target)] = predictions_test[(model_stacking, target)]

        # XGBoost matrices
        xg_train = xgb.DMatrix(X_train_target, label=y_log_train[target])
        xg_test = xgb.DMatrix(X_test_target, label=y_log_test[target])

        # Train
        # First stacking without any log:
        # param = {'silent': 1, 'nthread': 8, 'objective': 'reg:linear',
        #          'eta': 0.01, 'max_depth': 10, 'min_child_weight': 1, 'colsample_bytree': 1,
        #          'subsample': 0.75, 'gamma': 0, 'alpha': 12, 'lambda': 12, 'lambda_bias': 0}
        # Stacking predictions obtained with log:
        # param = {'silent': 1, 'nthread': 8, 'objective': 'reg:linear',
        #          'eta': 0.01, 'max_depth': 8, 'min_child_weight': 1, 'colsample_bytree': 1,
        #          'subsample': 0.5, 'gamma': 0, 'alpha': 12, 'lambda': 12, 'lambda_bias': 0}
        # Stacking predictions obtained with log and predicting log values
        param = {'silent': 1, 'nthread': 8, 'objective': 'reg:linear',
                 'eta': 0.01, 'max_depth': 8, 'min_child_weight': 1, 'colsample_bytree': 1,
                 'subsample': 0.5, 'gamma': 0, 'alpha': 1, 'lambda': 2, 'lambda_bias': 0}

        n_rounds = 10000
        model[target] = xgb.train(param, xg_train, n_rounds, [(xg_train, 'train'), (xg_test, 'test')],
                                  # feval=rmsle_evalerror,
                                  early_stopping_rounds=60)

        # Predict and clip values
        y_pred_target = model[target].predict(xg_test).clip(min=0)

        # # Scale data
        # scaler = preprocessing.StandardScaler()
        # X_train_target = scaler.fit_transform(X_train_target)
        # X_test_target = scaler.transform(X_test_target)
        #
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
        # early_stopping = EarlyStopping(monitor='val_loss', patience=8, verbose=0)
        # model.fit(X_train_target, y_train[target], validation_data=(X_test_target, y_test[target]), shuffle=True,
        #           callbacks=[early_stopping], nb_epoch=160, batch_size=16)

        # Predict, reshape and clip values
        # y_pred_target = model.predict(X_test_target).reshape(X_test_target.shape[0]).clip(min=0)

        # Transform results
        y_pred_target = np.exp(y_pred_target) - 1

        # Add target predictions
        y_pred += y_pred_target

        # Save best round
        best_round[target][i] = model[target].best_iteration

    # Evaluate
    rmsle_fold[i] = rmsle(y_test[COUNT], y_pred)
    print 'Fold %d/%d, RMSLE = %f, best it. = %d, %d' % (
        i + 1, n_folds, rmsle_fold[i], best_round[CASUAL][i], best_round[REGISTERED][i])
    # print 'Fold %d/%d, RMSLE = %f' % (i + 1, n_folds, rmsle_fold[i])
    i += 1

# Show results
print best_round
print 'mean RMSLE = %f, ' % rmsle_fold.mean()
print 'mean best it. = %f, %f' % (best_round[CASUAL].mean(), best_round[REGISTERED].mean())
