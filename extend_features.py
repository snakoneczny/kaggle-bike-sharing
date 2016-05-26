from utils import *
from datetime import datetime
from math import pi, sin, cos


def add_features(data_frame):
    print 'Adding date..'
    data_frame['year'] = data_frame.apply(lambda row: datetime.strptime(row['datetime'], '%Y-%m-%d %H:%M:%S').year,
                                          axis=1)
    data_frame['month'] = data_frame.apply(lambda row: datetime.strptime(row['datetime'], '%Y-%m-%d %H:%M:%S').month,
                                           axis=1)
    data_frame['day'] = data_frame.apply(lambda row: datetime.strptime(row['datetime'], '%Y-%m-%d %H:%M:%S').day,
                                         axis=1)
    data_frame['hour'] = data_frame.apply(lambda row: datetime.strptime(row['datetime'], '%Y-%m-%d %H:%M:%S').hour,
                                          axis=1)
    data_frame['weekday'] = data_frame.apply(
        lambda row: datetime.strptime(row['datetime'], '%Y-%m-%d %H:%M:%S').weekday(), axis=1)

    print 'Transforming seasons..'
    new_seasons = {1: 3, 2: 4, 3: 2, 4: 1}
    data_frame['season_ordered'] = data_frame.apply(lambda row: new_seasons[row['season']], axis=1)

    print 'Adding circular features..'
    data_frame['month_sin'] = data_frame.apply(lambda row: sin(((row['month'] - 5) % 12) / 12.0 * 2 * pi), axis=1)
    data_frame['month_cos'] = data_frame.apply(lambda row: cos(((row['month'] - 5) % 12) / 12.0 * 2 * pi), axis=1)
    data_frame['day_sin'] = data_frame.apply(lambda row: sin(row['day'] / 31.0 * 2 * pi), axis=1)
    data_frame['day_cos'] = data_frame.apply(lambda row: cos(row['day'] / 31.0 * 2 * pi), axis=1)
    data_frame['hour_sin'] = data_frame.apply(lambda row: sin(row['hour'] / 24.0 * 2 * pi), axis=1)
    data_frame['hour_cos'] = data_frame.apply(lambda row: cos(row['hour'] / 24.0 * 2 * pi), axis=1)
    data_frame['hour_sin'] = data_frame.apply(lambda row: sin(row['hour'] / 24.0 * 2 * pi), axis=1)
    data_frame['hour_cos'] = data_frame.apply(lambda row: cos(row['hour'] / 24.0 * 2 * pi), axis=1)
    data_frame['season_sin'] = data_frame.apply(lambda row: sin(((row['season'] - 3) % 4) / 4.0 * 2 * pi), axis=1)
    data_frame['season_cos'] = data_frame.apply(lambda row: cos(((row['season'] - 3) % 4) / 4.0 * 2 * pi), axis=1)
    data_frame['weekday_sin'] = data_frame.apply(lambda row: sin(row['weekday'] / 7.0 * 2 * pi), axis=1)
    data_frame['weekday_cos'] = data_frame.apply(lambda row: cos(row['weekday'] / 7.0 * 2 * pi), axis=1)

    # Drop datetime
    data_frame.drop('datetime', axis=1, inplace=True)


# Read data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Adding new features
add_features(train)
add_features(test)

# Save new data
train.to_csv('data/train_extended.csv', index=False)
test.to_csv('data/test_extended.csv', index=False)
