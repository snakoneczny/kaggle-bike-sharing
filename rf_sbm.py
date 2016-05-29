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

# Define targets
targets = [CASUAL, REGISTERED]
y_pred = {COUNT: np.zeros(X_test.shape[0]), CASUAL: np.zeros(X_test.shape[0]), REGISTERED: np.zeros(X_test.shape[0])}

# Train
clf = {}
for target in targets:
    clf[target] = RandomForestRegressor(random_state=0, n_jobs=8, n_estimators=400, max_features=None,
                                        max_depth=None, min_samples_split=1)
    clf[target].fit(X_train, y_train[target])

# Predict and clip values lower than 0
for target in targets:
    y_pred[target] = clf[target].predict(X_test).clip(min=0)
    y_pred[COUNT] += y_pred[target]

# Write submission
write_submission(y_pred, 'submissions/rf_%s.csv' % features)
write_submission_stacking(y_pred, 'submissions/rf_%s_stacking.csv' % features)
