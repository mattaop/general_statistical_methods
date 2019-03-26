import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import Embedding, Dense, Input, concatenate, Flatten, Dropout, BatchNormalization
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from sklearn.model_selection import train_test_split
import keras.backend as K


def load_data(samples):
    # get data from csv files
    data = pd.read_csv('data/train.csv', usecols=['title', 'description', 'region', 'city', 'parent_category_name',
                                                  'category_name', 'price', 'deal_probability'])
    y = data['deal_probability']
    x = data.drop('deal_probability', axis=1)
    x = x.iloc[:samples, ]
    y = y.iloc[:samples, ]
    return x, y


def fill_nans(data):
    cols = ['title', 'description', 'region', 'city', 'parent_category_name', 'category_name']
    for c in cols:
        data[c].fillna(" ", inplace=True)
    data['price'].fillna(value=-1, inplace=True)
    return data


def encode_categorical_data(data):
    cat_cols = ['region', 'city', 'parent_category_name', 'category_name']

    le_encoders = {x: LabelEncoder() for x in cat_cols}
    label_enc_cols = {k: v.fit_transform(data[k]) for k, v in le_encoders.items()}

    return le_encoders, label_enc_cols


def transform_data(data, max_features, max_title, max_desc):
    cols = ['title', 'description', 'region', 'city', 'parent_category_name', 'category_name']
    for c in cols:
        data[c] = [one_hot(d, max_features) for d in data[c]]

    title = np.asarray(data['title'])
    desc = np.asarray(data['description'])
    region = np.asarray(data['region'])
    city = np.asarray(data['city'])
    cat1 = np.asarray(data['parent_category_name'])
    cat2 = np.asarray(data['category_name'])

    #title = [one_hot(d, max_features) for d in title]
    #desc = [one_hot(d, max_features) for d in desc]
    #region = [one_hot(d, max_features) for d in region]
    #city = [one_hot(d, max_features) for d in city]
    #cat1 = [one_hot(d, max_features) for d in cat1]
    #cat2 = [one_hot(d, max_features) for d in x_cat2]

    title = pad_sequences(title, maxlen=max_title, padding='post')
    desc = pad_sequences(desc, maxlen=max_desc, padding='post')
    region = pad_sequences(region, maxlen=5, padding='post')
    city = pad_sequences(city, maxlen=5, padding='post')
    cat1 = pad_sequences(cat1, maxlen=10, padding='post')
    cat2 = pad_sequences(cat2, maxlen=10, padding='post')

    stdScaler = StandardScaler(with_mean=0, with_std=1)
    data[['price']] = stdScaler.fit_transform(data[['price']])
    price = np.asarray(data['price'])
    print(price)

    return title, desc, region, city, cat1, cat2, price


def build_model(max_features, max_title, max_desc):
    # define two sets of inputs
    x_title = Input(shape=(max_title,))
    x_desc = Input(shape=(max_desc,))
    x_region = Input(shape=(5,))
    x_city = Input(shape=(5,))
    x_cat1 = Input(shape=(10,))
    x_cat2 = Input(shape=(10,))
    x_price = Input(shape=(1,))

    # the first network operating on title
    embedding_layer = Embedding(input_dim=max_features, output_dim=32)

    x1 = embedding_layer(x_title)
    x1 = Dense(32, activation='sigmoid')(x1)
    x1 = Flatten()(x1)
    x1 = Dense(1, activation='sigmoid')(x1)

    x2 = embedding_layer(x_desc)
    x2 = Dense(32, activation='sigmoid')(x2)
    x2 = Flatten()(x2)
    x2 = Dense(1, activation='sigmoid')(x2)

    # the third layer containing categorical values (region, city, category and price)
    y1 = Embedding(input_dim=1, output_dim=4)(x_region)
    y1 = Flatten()(y1)
    y1 = Dense(1, activation='relu')(y1)

    y2 = Embedding(input_dim=1, output_dim=4)(x_city)
    y2 = Flatten()(y2)
    y2 = Dense(1, activation='relu')(y2)

    y3 = Embedding(input_dim=1, output_dim=4)(x_cat1)
    y3 = Flatten()(y3)
    y3 = Dense(1, activation='relu')(y3)

    y4 = Embedding(input_dim=1, output_dim=4)(x_cat2)
    y4 = Flatten()(y4)
    y4 = Dense(1, activation='relu')(y4)

    y5 = Dense(1, activation='relu')(x_price)

    y = concatenate([y1, y2, y3, y4, y5])
    y = Dense(1, activation='sigmoid')(y)

    # combine the outputs
    z = concatenate([x1, x2, y])
    z = BatchNormalization()(z)
    z = Dense(3, activation="relu")(z)
    z = Dense(1, activation="linear")(z)

    # define inputs and outputs
    model = Model(inputs=[x_title, x_desc, x_region, x_city, x_cat1, x_cat2, x_price], outputs=z)

    return model


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def main():
    samples = 30000
    max_features = 1000
    max_title = 30
    max_desc = 100

    print("Loading data...")
    data, y = load_data(samples)

    print("Processing data...")
    data = fill_nans(data)

    #train_x, test_x, train_y, test_y = train_test_split(data, y, test_size=0.25, random_state=23)
    x_title, x_desc, x_region, x_city, x_cat1, x_cat2, x_price = transform_data(data, max_features, max_title, max_desc)

    print(x_title)

    print("Building model...")
    model = build_model(max_features, max_title, max_desc)
    model.compile(optimizer='adam', loss=root_mean_squared_error, metrics=[root_mean_squared_error])
    model.fit([x_title, x_desc, x_region, x_city, x_cat1, x_cat2, x_price], y,
              epochs=15, verbose=True, batch_size=32, validation_split=0.2)


main()
