import load_data as Ld
import NeuralNetwork as Nn
from keras.datasets import imdb
from keras.preprocessing import sequence


def fit_rnn(x_train, y_train, x_test, y_test, max_features, max_length):
    x_train = sequence.pad_sequences(x_train, maxlen=max_length)
    x_test = sequence.pad_sequences(x_test, maxlen=max_length)
    model_nn = Nn.NeuralNetwork('ffn', x_train, y_train, max_features, max_length)
    model_nn.train(x_train, y_train, x_test, y_test)


def main():
    max_features = 5000
    max_length = 500
    print("Loading data...")
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    print("Fitting model...")
    fit_rnn(x_train, y_train, x_test, y_test, max_features, max_length)


main()
