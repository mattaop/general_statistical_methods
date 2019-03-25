from keras.models import Sequential,Input,Model
from keras.layers import Input, Dense, Embedding, SimpleRNN, LSTM, Conv1D, Dropout, MaxPooling1D, Flatten, concatenate
import matplotlib.pyplot as plt
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
        x_region = Input(shape=(self.max_input_length,))
        x_city = Input(shape=(self.max_input_length,))
        x_cat1 = Input(shape=(self.max_input_length,))
        x_cat2 = Input(shape=(self.max_input_length,))
        x_price = Input(shape=(1,))

        # the first network operating on title
        x1 = Embedding(input_dim=self.max_features, output_dim=32, input_length=self.max_input_length)(x_title)
        x1 = LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(x1)
        x1 = LSTM(128, dropout=0.2, recurrent_dropout=0.2)(x1)
        x1 = Dense(1, activation='sigmoid')(x1)

        # the second network operating on description
        x2 = Embedding(input_dim=self.max_features, output_dim=32, input_length=self.max_input_length)(x_desc)
        x2 = LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(x2)
        x2 = LSTM(128, dropout=0.2, recurrent_dropout=0.2)(x2)
        x2 = Dense(1, activation='sigmoid')(x2)

        # the third layer containing categorical values (region, city, category and price)
        y1 = Embedding(input_dim=self.max_features, output_dim=32, input_length=self.max_input_length)(x_region)
        y1 = Flatten()(y1)
        y1 = Dense(1, activation='relu')(y1)

        y2 = Embedding(input_dim=self.max_features, output_dim=32, input_length=self.max_input_length)(x_city)
        y2 = Flatten()(y2)
        y2 = Dense(1, activation='relu')(y2)

        y3 = Embedding(input_dim=self.max_features, output_dim=32, input_length=self.max_input_length)(x_cat1)
        y3 = Flatten()(y3)
        y3 = Dense(1, activation='relu')(y3)

        y4 = Embedding(input_dim=self.max_features, output_dim=32, input_length=self.max_input_length)(x_cat2)
        y4 = Flatten()(y4)
        y4 = Dense(1, activation='relu')(y4)

        y5 = Dense(1, activation='relu')(x_price)

        y = concatenate([y1, y2, y3, y4, y5])
        y = Dense(16, activation='relu')(y)
        y = Dense(1, activation='sigmoid')(y)

        # combine the output of the two branches
        combined = concatenate([x1, x2, y])

        # apply a FC layer and then a regression prediction on the combined outputs
        z = Dense(3, activation="relu")(combined)
        z = Dense(1, activation="linear")(z)

        # define inputs and outputs
        model = Model(inputs=[x_title, x_desc, x_region, x_city, x_cat1, x_cat2, x_price], outputs=z)

        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        model.summary()

        return model

    def train(self, x_title, x_desc, x_region, x_city, x_cat1, x_cat2, x_price, y, batch_size=32, epochs=5, verbose=True):
        history = self.network.fit([x_title, x_desc, x_region, x_city, x_cat1, x_cat2, x_price], y, epochs=epochs,
                                   verbose=verbose, batch_size=batch_size, validation_split=0.3)
        #(loss, accuracy) = self.network.evaluate(x_train, y_train)
        #print("Training Accuracy: {:.4f}".format(accuracy))
        #(loss, accuracy) = self.network.evaluate(x_test, y_test)
        #print("Testing Accuracy:  {:.4f}".format(accuracy))
        self.plot_results(history)

    def plot_results(self, history):
        # show the terrible predictions
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()
