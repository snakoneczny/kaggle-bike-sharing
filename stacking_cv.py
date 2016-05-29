from utils import *
from sklearn.cross_validation import KFold
import xgboost as xgb

# Parameters
features = EXTENDED

# Read data
train = pd.read_csv('data/train_%s.csv' % features)

# Get X and y
X = train.drop([CASUAL, REGISTERED, COUNT], inplace=False, axis=1)
y = train[[CASUAL, REGISTERED, COUNT]]

# Define targets
targets = [CASUAL, REGISTERED]

# Read predictions from previous models
models_stacking = ['rf_extended', 'xgb_extended', 'keras_neuralnet']
predictions_to_stack = pd.DataFrame()
for model_stacking in models_stacking:
    predictions_model = pd.read_csv('cross-validation/%s.csv' % model_stacking)
    for target in targets:
        predictions_to_stack[(model_stacking, target)] = predictions_model[target]

# CV
n_folds = 10
rmsle_fold = np.zeros(n_folds)
best_round = {CASUAL: np.zeros(n_folds), REGISTERED: np.zeros(n_folds)}
skf = KFold(y.shape[0], n_folds, shuffle=True, random_state=0)
i = 0
for train, test in skf:
    X_train, X_test = X.loc[train], X.loc[test]
    y_train, y_test = y.loc[train], y.loc[test]
    predictions_train, predictions_test = predictions_to_stack.loc[train], predictions_to_stack.loc[test]

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
        xg_train = xgb.DMatrix(X_train_target, label=y_train[target])
        xg_test = xgb.DMatrix(X_test_target, label=y_test[target])

        # Train
        # param = {'silent': 1, 'nthread': 8, 'objective': 'reg:linear',
        #          'eta': 0.01, 'max_depth': 10, 'min_child_weight': 2, 'colsample_bytree': 1,
        #          'subsample': 0.5, 'gamma': 0, 'alpha': 2, 'lambda': 2, 'lambda_bias': 0}
        param = {'silent': 1, 'nthread': 8, 'objective': 'reg:linear',
                 'eta': 0.01, 'max_depth': 10, 'min_child_weight': 1, 'colsample_bytree': 1,
                 'subsample': 0.75, 'gamma': 0, 'alpha': 2, 'lambda': 2, 'lambda_bias': 0}
        n_rounds = 2000
        model[target] = xgb.train(param, xg_train, n_rounds, [(xg_train, 'train'), (xg_test, 'test')],
                                  feval=rmsle_evalerror, early_stopping_rounds=60)

        # Predict and clip values
        y_pred_target = model[target].predict(xg_test).clip(min=0)
        y_pred += y_pred_target

        # Save best round
        best_round[target][i] = model[target].best_iteration

    # Evaluate
    rmsle_fold[i] = rmsle(y_test['count'], y_pred)
    print 'Fold %d/%d, RMSLE = %f, best it. = %d, %d' % (
        i + 1, n_folds, rmsle_fold[i], model[CASUAL].best_iteration, model[REGISTERED].best_iteration)
    i += 1

# Show results
print best_round
print 'mean RMSLE = %f, ' % rmsle_fold.mean()
print 'mean best it. = %f, %f' % (best_round[CASUAL].mean(), best_round[REGISTERED].mean())
