from utils import *
from sklearn.ensemble import RandomForestRegressor

# Parameters
features = EXTENDED

# Read data
train = pd.read_csv('data/train_%s.csv' % features)
test = pd.read_csv('data/test_%s.csv' % features)

# Get X and y
X_train = train.drop([CASUAL, REGISTERED, COUNT], inplace=False, axis=1)
y_train = train[[CASUAL, REGISTERED]]
X_test = test

# Transform y
y_log_train = np.log(y_train + 1)

# Define targets
targets = [CASUAL, REGISTERED]
y_pred = {COUNT: np.zeros(X_test.shape[0]), CASUAL: np.zeros(X_test.shape[0]), REGISTERED: np.zeros(X_test.shape[0])}

# Train
model = {}
for target in targets:
    model[target] = RandomForestRegressor(random_state=0, n_jobs=8, n_estimators=400, max_features=None,
                                          max_depth=None, min_samples_split=1)
    model[target].fit(X_train, y_log_train[target])

# Predict
for target in targets:
    y_pred[target] = model[target].predict(X_test).clip(min=0)
    y_pred[target] = np.exp(y_pred[target]) - 1
    y_pred[COUNT] += y_pred[target]

# Write submission
write_results(y_pred, 'submissions/rf_%s.csv' % features, 'data/test.csv')
write_results_extended(y_pred, 'submissions/rf_%s_stacking.csv' % features, 'data/test.csv')
