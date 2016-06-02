from utils import *
from sklearn import preprocessing

np.random.seed(1227)

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

# Parameters
features = NEURAL_NET

# Read data
train = pd.read_csv('data/train_%s.csv' % features)
test = pd.read_csv('data/test_%s.csv' % features)

# Get X and y
X_train = train.drop(['casual', 'registered', 'count',
                      'month', 'day', 'season', 'weekday',
                      'weather', 'humidity_inv', 'windspeed_inv',
                      ], inplace=False, axis=1)
X_test = test.drop(['month', 'day', 'season', 'weekday',
                    'weather', 'humidity_inv', 'windspeed_inv',
                    ], inplace=False, axis=1)
y_train = train[[CASUAL, REGISTERED]]

# Transform y
# y_log_train = np.log(y_train + 1)

# Scale data
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define targets
targets = [CASUAL, REGISTERED]
y_pred = {COUNT: np.zeros(X_test.shape[0]), CASUAL: np.zeros(X_test.shape[0]), REGISTERED: np.zeros(X_test.shape[0])}

# Work with targets
for target in targets:
    y_train_target = y_train[target].as_matrix()
    # y_log_train_target = y_log_train[target].as_matrix()

    # Define neural network
    model = Sequential()
    model.add(Dense(200, input_dim=X_train.shape[1]))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(loss='mean_squared_logarithmic_error', optimizer='rmsprop')

    # Train
    model.fit(X_train, y_train_target, shuffle=True, nb_epoch=50, batch_size=16)

    # Predict
    y_pred[target] = model.predict(X_test).reshape(X_test.shape[0]).clip(min=0)
    # y_pred[target] = np.exp(y_pred[target]) - 1
    y_pred[COUNT] += y_pred[target]

# Write submission
write_results(y_pred, 'submissions/keras_%s.csv' % features, 'data/test.csv')
write_results_extended(y_pred, 'submissions/keras_%s_stacking.csv' % features, 'data/test.csv')
