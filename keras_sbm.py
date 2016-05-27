from utils import *
from sklearn import preprocessing

np.random.seed(1227)

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
from keras.regularizers import l2, activity_l2, l1

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
y_train = train[['casual', 'registered']]

# Scale data
scaler = preprocessing.StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define different targets
targets = ['casual', 'registered']

# Work with targets
y_pred = np.zeros(X_test.shape[0])
for target in targets:
    y_train_target = y_train[target].as_matrix()

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
    history = model.fit(X_train, y_train_target, shuffle=True, nb_epoch=50, batch_size=16)

    # Predict, reshape and clip values
    y_pred += model.predict(X_test).reshape(X_test.shape[0]).clip(min=0)

# Write submission
write_submission(y_pred, 'submissions/keras_%s.csv' % features)
