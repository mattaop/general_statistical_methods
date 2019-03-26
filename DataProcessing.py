from sklearn.preprocessing import StandardScaler
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
import sklearn.preprocessing


import pandas as pd
import numpy as np


class DataProcessing:
    def __init__(self, data, samples, max_title, max_desc, max_features, test=False):
        self.data = data
        self.samples = samples
        self.max_title = max_title
        self.max_desc = max_desc
        self.max_features = max_features
        self.test = test
        self.fill_nans()
        self.convert_date()
        self.shrink_data()
        self.process_data()
        self.title = (self.data['title'])
        self.desc = (self.data['description'])
        self.region = (self.data['region'])
        self.city = (self.data['city'])
        self.cat1 = (self.data['parent_category_name'])
        self.cat2 = (self.data['category_name'])
        self.date = self.data['activation_date']
        self.param1 = (self.data['param_1'])
        self.param2 = (self.data['param_2'])
        self.param3 = (self.data['param_3'])
        self.user_type = (self.data['user_type'])
        self.item_number = (self.data['item_seq_number'])
        self.price = (self.data['price'])
        self.y = (self.data['deal_probability'])
        self.pad_sequence()

    def fill_nans(self):
        cols = ['title', 'description', 'region', 'city', 'parent_category_name', 'category_name', 'param_1', 'param_2',
                'param_3', 'user_type']
        for c in cols:
            self.data[c].fillna(" ", inplace=True)
        self.data['item_seq_number'].fillna(value=-1, inplace=True)
        self.data['price'].fillna(value=-1, inplace=True)
        self.data['activation_date'].fillna(value=-1, inplace=True)

    def shrink_data(self):
        if self.test:
            self.data = self.data.iloc[-int(len(self.data['deal_probability'].values)*0.25):, ]
        else:
            self.data = self.data.iloc[:self.samples, ]

    def process_data(self):
        i = -1
        cols = ['title', 'description', 'region', 'city', 'parent_category_name', 'category_name', 'param_1', 'param_2',
                'param_3', 'user_type']
        for c in cols:
            i += 1
            self.data[c] = [one_hot(d, self.max_features[i]) for d in self.data[c]]

    def pad_sequence(self):
        self.title = pad_sequences(self.title, maxlen=self.max_title, padding='post')
        self.desc = pad_sequences(self.desc, maxlen=self.max_desc, padding='post')
        self.region = pad_sequences(self.region, maxlen=5, padding='post')
        self.city = pad_sequences(self.city, maxlen=5, padding='post')
        self.cat1 = pad_sequences(self.cat1, maxlen=10, padding='post')
        self.cat2 = pad_sequences(self.cat2, maxlen=10, padding='post')
        self.param1 = pad_sequences(self.param1, maxlen=5, padding='post')
        self.param2 = pad_sequences(self.param2, maxlen=5, padding='post')
        self.param3 = pad_sequences(self.param3, maxlen=5, padding='post')
        self.user_type = pad_sequences(self.user_type, maxlen=3, padding='post')

        scale = StandardScaler(with_mean=0, with_std=1)
        scale.fit(self.price.values.reshape(-1, 1))
        self.price = scale.transform(self.price.values.reshape(-1, 1))
        scale.fit(self.item_number.values.reshape(-1, 1))
        self.price = scale.transform(self.item_number.values.reshape(-1, 1))

    def feature_extraction(self):
        pass

    def convert_date(self):
        self.data['activation_date'] = np.expand_dims(
            pd.to_datetime(self.data['activation_date']).dt.weekday.astype(np.int32).values, axis=-1)

