from utils import *
import xgboost as xgb

# Parameters
features = EXTENDED

# Read data
train = pd.read_csv('data/train_%s.csv' % features)
test = pd.read_csv('data/test_%s.csv' % features)

# Get X and y
X_train = train.drop([COUNT, CASUAL, REGISTERED], inplace=False, axis=1)
X_test = test
y_train = train[[COUNT, CASUAL, REGISTERED]]

# Define different targets
targets = [CASUAL, REGISTERED]
y_pred = {COUNT: np.zeros(X_test.shape[0]), CASUAL: np.zeros(X_test.shape[0]),
          REGISTERED: np.zeros(X_test.shape[0])}
n_rounds = {CASUAL: 260, REGISTERED: 350}

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

    # XGBoost matrices
    xg_train = xgb.DMatrix(X_train_target.as_matrix(), label=y_train[target].as_matrix())
    xg_test = xgb.DMatrix(X_test_target.as_matrix())

    # Train
    param = {'silent': 1, 'nthread': 8, 'objective': 'reg:linear',
             'eta': 0.01, 'max_depth': 10, 'min_child_weight': 2, 'colsample_bytree': 1,
             'subsample': 0.5, 'gamma': 0, 'alpha': 2, 'lambda': 2, 'lambda_bias': 0}
    model = xgb.train(param, xg_train, n_rounds[target], [(xg_train, 'train')], feval=rmsle_evalerror)

    # Predict and clip
    y_pred[target] = model.predict(xg_test).clip(min=0)
    y_pred[COUNT] += y_pred[target]

# Write submission
write_submission(y_pred, 'submissions/stacking_xgb_%s.csv' % features, 'data/test.csv')
