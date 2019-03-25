import load_data as Ld
import NeuralNetwork as Nn
from keras.datasets import imdb
from keras.preprocessing import sequence
import matplotlib.pyplot as plt


def fit_rnn(x_title, x_desc, y, max_features, max_length):
    x_title = sequence.pad_sequences(x_title, maxlen=max_length)
    x_desc = sequence.pad_sequences(x_desc, maxlen=max_length)
    model_nn = Nn.NeuralNetwork('combined_network', max_features, max_length)
    model_nn.train(x_title, x_desc, y)


def main():
    max_features = 5000
    max_length = 500
    print("Loading data...")
    #(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    #print(x_train[1])
    x_title, x_desc, y = Ld.load_data(max_length)
    print("Fitting model...")
    fit_rnn(x_title, x_desc, y, max_features, max_length)


main()
