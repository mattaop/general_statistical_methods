import load_data as Ld
import NeuralNetwork as Nn
from keras.datasets import imdb
import matplotlib.pyplot as plt


def main():
    max_features = 5000
    max_length = 500
    print("Loading data...")
    #(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    #print(x_train[1])
    x_title, x_desc, x_region, x_city, x_cat1, x_cat2, x_price, y = Ld.load_data(max_features, max_length)
    print("Fitting model...")
    model_nn = Nn.NeuralNetwork('combined_network', max_features, max_length)
    model_nn.train(x_title, x_desc, x_region, x_city, x_cat1, x_cat2, x_price, y)


main()
