from utils import *
from sklearn.ensemble import RandomForestRegressor

# Parameters
features = EXTENDED

# Read data
train = pd.read_csv('data/train_%s.csv' % features)
test = pd.read_csv('data/test_%s.csv' % features)

# Get X and y
X_train = train.drop(['casual', 'registered', 'count'], inplace=False, axis=1)
y_train = train[['casual', 'registered']]
X_test = test

# Train
targets = ['casual', 'registered']
clf = {}
for target in targets:
    clf[target] = RandomForestRegressor(random_state=0, n_jobs=8, n_estimators=400, max_features=None,
                                        max_depth=None, min_samples_split=1)
    clf[target].fit(X_train, y_train[target])

# Predict
y_pred = clf['casual'].predict(X_test) + clf['registered'].predict(X_test)

# Clip values lower than 0
y_pred = y_pred.clip(min=0)

# Write submission
write_submission(y_pred, 'submissions/rf_%s.csv' % features)