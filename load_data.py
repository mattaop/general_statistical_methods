import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def load_data(max_features):
    T = 30000

    # get data from csv files
    data = pd.read_csv('data/train.csv', usecols=['description', 'title', 'deal_probability'])
    x_title = (data['title'])
    x_desc = (data['description'])
    y = (data['deal_probability'])

    # break up data into train and test data
    #x_title_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=23)

    # shrink training data to T
    x_title = x_title[:T]
    x_desc = x_desc[:T]
    y = y[:T]

    # Replace nans with spaces
    x_title.fillna(" ", inplace=True)
    x_desc.fillna(" ", inplace=True)

    vec = TfidfVectorizer(max_features=max_features)
    x_title = vec.fit_transform(x_title).toarray()
    x_desc = vec.transform(x_desc).toarray()

    return x_title, x_desc, y
