import pandas as pd
from sklearn.model_selection import train_test_split


def load_data():
    T = 30000
    mdf = 50

    # get data from csv files
    data = pd.read_csv('data/train.csv', usecols=['description', 'deal_probability'])
    x = (data['description'])
    y = (data['deal_probability'])

    # break up data into train and test data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=23)

    # shrink training data to T
    x_train = x_train[:T]
    y_train = y_train[:T]

    # Replace nans with spaces
    x_train.fillna(" ", inplace=True)
    x_test.fillna(" ", inplace=True)

    return x_train, y_train, x_test, y_test
