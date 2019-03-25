from keras.models import Sequential,Input,Model
from keras.layers import Input, Dense, Embedding, SimpleRNN, LSTM, Conv1D, Dropout, MaxPooling1D, Flatten, concatenate
import matplotlib.pyplot as plt
from keras.utils import plot_model
import numpy as np


class NeuralNetwork:
    def __init__(self, model_type, max_features=5000, max_input_length=500):
        self.batch_size = 32
        self.epochs = 15
        self.max_features = max_features
        self.max_input_length = max_input_length
        if model_type == 'rnn1':
            self.network = self.rnn1()
        elif model_type == 'rnn2':
            self.network = self.rnn2()
        elif model_type == 'cnn':
            self.network = self.cnn1D()
        elif model_type == 'combined_network':
            self.network = self.combined_network()
        else:
            self.network = self.ffn()

    def ffn(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.max_features, output_dim=32, input_length=self.max_input_length))
        model.add(Flatten())
        model.add(Dense(250, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model

    def cnn1D(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.max_features, output_dim=32, input_length=self.max_input_length))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(250, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        return model

    def rnn1(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.max_features, output_dim=32, input_length=self.max_input_length))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model

    def rnn2(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.max_features, output_dim=32, input_length=self.max_input_length))
        model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
        model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
        model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model

    def combined_network(self):
        # define two sets of inputs
        x_title = Input(shape=(self.max_input_length,))
        x_desc = Input(shape=(self.max_input_length,))

        # the first network
        x = Embedding(input_dim=self.max_features, output_dim=32, input_length=self.max_input_length)(x_title)
        x = LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(x)
        x = LSTM(128, dropout=0.2, recurrent_dropout=0.2)(x)
        x = Dense(1, activation='sigmoid')(x)

        # the second branch operates on the second input
        y = Embedding(input_dim=self.max_features, output_dim=32, input_length=self.max_input_length)(x_desc)
        y = LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(y)
        y = LSTM(128, dropout=0.2, recurrent_dropout=0.2)(y)
        y = Dense(1, activation='sigmoid')(y)

        # combine the output of the two branches
        combined = concatenate([x, y])

        # apply a FC layer and then a regression prediction on the
        # combined outputs
        z = Dense(2, activation="relu")(combined)
        z = Dense(1, activation="linear")(z)

        # our model will accept the inputs of the two branches and
        # then output a single value
        model = Model(inputs=[x_title, x_desc], outputs=z)

        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        #plot_model(model, to_file='model.png')
        model.summary()
        return model

    def train(self, x_title, x_desc, y, batch_size=32, epochs=5, verbose=True):
        history = self.network.fit([x_title, x_desc], y, epochs=epochs, verbose=verbose, batch_size=batch_size, validation_split=0.3)
        #(loss, accuracy) = self.network.evaluate(x_train, y_train)
        #print("Training Accuracy: {:.4f}".format(accuracy))
        #(loss, accuracy) = self.network.evaluate(x_test, y_test)
        #print("Testing Accuracy:  {:.4f}".format(accuracy))
        self.plot_results(history)

    def plot_results(self, history):
        # show the terrible predictions
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

