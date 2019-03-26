import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.sequence import pad_sequences
import DataProcessing as Dp
from keras.preprocessing.text import one_hot
from collections import Counter
import numpy as np


def load_data(samples, max_title, max_desc, max_features_text, max_features_region, max_features_city,
              max_features_parent_category_name, max_features_category_name, test=False):
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
    if test:
        x_title = x_title[1127569:]
        x_desc = x_desc[1127569:]
        x_region = x_region[1127569:]
        x_city = x_city[1127569:]
        x_cat1 = x_cat1[1127569:]
        x_cat2 = x_cat2[1127569:]
        x_price = x_price[1127569:]
        y = y[1127569:]
    else:
        x_title = x_title[:samples]
        x_desc = x_desc[:samples]
        x_region = x_region[:samples]
        x_city = x_city[:samples]
        x_cat1 = x_cat1[:samples]
        x_cat2 = x_cat2[:samples]
        x_price = x_price[:samples]
        y = y[:samples]

    # Replace nans with spaces
    x_title.fillna(" ", inplace=True)
    x_desc.fillna(" ", inplace=True)
    x_region.fillna(" ", inplace=True)
    x_city.fillna(" ", inplace=True)
    x_cat1.fillna(" ", inplace=True)
    x_cat2.fillna(" ", inplace=True)

    # Replace nan with 0
    x_price = np.nan_to_num(x_price)

    # Process text data
    #text_processing = Dp.DataProcessing()
    #x_title = text_processing.process_data(x_title)
    #x_desc = text_processing.process_data(x_desc)

    x_title = [one_hot(d, max_features_text) for d in x_title]
    x_desc = [one_hot(d, max_features_text) for d in x_desc]
    x_region = [one_hot(d, max_features_region) for d in x_region]
    x_city = [one_hot(d, max_features_city) for d in x_city]
    x_cat1 = [one_hot(d, max_features_parent_category_name) for d in x_cat1]
    x_cat2 = [one_hot(d, max_features_category_name) for d in x_cat2]
    #x_region = vec.transform(x_region).toarray()
    #x_city = vec.transform(x_city).toarray()
    #x_cat1 = vec.transform(x_cat1).toarray()
    #x_cat2 = vec.transform(x_cat2).toarray()


    x_title = pad_sequences(x_title, maxlen=max_title, padding='post')
    x_desc = pad_sequences(x_desc, maxlen=max_desc, padding='post')
    x_region = pad_sequences(x_region, maxlen=5, padding='post')
    x_city = pad_sequences(x_city, maxlen=5, padding='post')
    x_cat1 = pad_sequences(x_cat1, maxlen=10, padding='post')
    x_cat2 = pad_sequences(x_cat2, maxlen=10, padding='post')

    scale = StandardScaler(with_mean=0, with_std=1)
    scale.fit(x_price.reshape(-1, 1))
    x_price = scale.transform(x_price.reshape(-1, 1))

    return x_title, x_desc, x_region, x_city, x_cat1, x_cat2, x_price, y
