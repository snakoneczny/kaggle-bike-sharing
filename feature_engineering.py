from utils import *
from datetime import datetime


# Parameters
engineering_type = SIMPLE


def feature_engineering(data_frame):

    if engineering_type == SIMPLE:
        process_seasons_simple(data_frame)
        process_datetime_simple(data_frame)


def process_seasons_simple(data_frame):
    new_seasons = {1: 3, 2: 4, 3: 2, 4: 1}
    data_frame['season'] = data_frame.apply(lambda row: new_seasons[row['season']], axis=1)


def process_datetime_simple(data_frame):

    pd.concat([data_frame, pd.DataFrame(columns=['year', 'month', 'day', 'hour'])])

    for i in xrange(data_frame.shape[0]):
        if not i % 100:
            print 'Processing row %d/%d' % (i, data_frame.shape[0])

        date = datetime.strptime(data_frame['datetime'][i], '%Y-%m-%d %H:%M:%S')

        data_frame.loc[i, 'year'] = date.year
        data_frame.loc[i, 'month'] = date.month
        data_frame.loc[i, 'day'] = date.day
        data_frame.loc[i, 'hour'] = date.hour

    data_frame.drop('datetime', axis=1, inplace=True)


# Read data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Feature engineering
feature_engineering(train)
feature_engineering(test)

# Save new data
train.to_csv('data/train_%s.csv' % engineering_type, index=False)
test.to_csv('data/test_%s.csv' % engineering_type, index=False)
