import DataProcessing as Dp
import NeuralNetwork as Nn
import pandas as pd
import numpy as np

samples = 1500000
max_title = 30
max_desc = 300
max_features_text = 50000
max_features_region = 28 + 1  # Number of regions
max_features_city = 1022 + 1  # Number of cities
max_features_parent_category_name = 9 + 1  # Parent categories
max_features_category_name = 47 + 1  # Categories
max_features_param1 = 500
max_features_param2 = 500
max_features_param3 = 500
max_features_img = 3066 + 1
max_features_user = 3 + 1  # Private, shop or company
max_features_date = 7 + 1  # Days in the week


max_features = [max_features_text, max_features_text, max_features_region, max_features_city,
                max_features_parent_category_name, max_features_category_name, max_features_param1,
                max_features_param2, max_features_param3, max_features_user, max_features_date, max_features_img]


def load_data():
    df_data = pd.read_csv('data/train.csv', usecols=['description', 'title', 'region', 'city', 'parent_category_name',
                                                     'category_name', 'price', 'activation_date', 'param_1', 'param_2',
                                                     'param_3', 'image_top_1', 'user_type', 'item_seq_number',
                                                     'deal_probability'])
    return df_data


def load_embedding_vectors(fname='cc.ru.300.vec.gz'):
    # en_model = FastText.load_fasttext_format('fname')
    pass


if __name__ == "__main__":
    print("Loading data...")
    df = load_data()

    print("Processing training data...")
    data_train = Dp.DataProcessing(df, samples, max_title, max_desc, max_features)

    print("Processing test data...")
    data_test = Dp.DataProcessing(df, samples, max_title, max_desc, max_features, test=True)

    # print("Load embedding vector...")
    # embedding_vector = load_embedding_vectors()

    print(data_train)
    print("Fitting model...")
    model = Nn.NeuralNetwork(max_title, max_desc, max_features)
    model.train(data_train, data_test)

    print("Testing model...")
    model.test(data_test)

    print("Saving weights...")
    model.save_weight()



