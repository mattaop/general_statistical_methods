import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from keras.preprocessing import sequence
import numpy as np


def load_data(max_features, max_length):
    T = 30000

    # get data from csv files
    data = pd.read_csv('data/train.csv', usecols=['description', 'title', 'region', 'city', 'parent_category_name',
                                                  'category_name', 'price', 'deal_probability'])
    x_title = (data['title'])
    x_desc = (data['description'])
    x_region = (data['region'])
    x_city = (data['city'])
    x_cat1 = (data['parent_category_name'])
    x_cat2 = (data['category_name'])
    x_price = (data['price'])

    y = (data['deal_probability'])

    # break up data into train and test data
    #x_title_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=23)

    # shrink training data to T
    x_title = x_title[:T]
    x_desc = x_desc[:T]
    x_region = x_region[:T]
    x_city = x_city[:T]
    x_cat1 = x_cat1[:T]
    x_cat2 = x_cat2[:T]
    x_price = x_price[:T]
    y = y[:T]

    # Replace nans with spaces
    x_title.fillna(" ", inplace=True)
    x_desc.fillna(" ", inplace=True)
    x_price = np.nan_to_num(x_price)

    vec = TfidfVectorizer(max_features=max_features)
    x_title = vec.fit_transform(x_title).toarray()
    x_desc = vec.transform(x_desc).toarray()
    x_region = vec.transform(x_region).toarray()
    x_city = vec.transform(x_city).toarray()
    x_cat1 = vec.transform(x_cat1).toarray()
    x_cat2 = vec.transform(x_cat2).toarray()

    x_title = sequence.pad_sequences(x_title, maxlen=max_length)
    x_desc = sequence.pad_sequences(x_desc, maxlen=max_length)
    x_region = sequence.pad_sequences(x_region, maxlen=max_length)
    x_city = sequence.pad_sequences(x_city, maxlen=max_length)
    x_cat1 = sequence.pad_sequences(x_cat1, maxlen=max_length)
    x_cat2 = sequence.pad_sequences(x_cat2, maxlen=max_length)

    scale = StandardScaler(with_mean=0, with_std=1)
    scale.fit(x_price.reshape(-1, 1))
    x_price = scale.transform(x_price.reshape(-1, 1))

    return x_title, x_desc, x_region, x_city, x_cat1, x_cat2, x_price, y
