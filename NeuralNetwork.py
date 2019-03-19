from keras.models import Sequential,Input,Model
from keras.layers import Input, Dense, Embedding, SimpleRNN, LSTM, Conv1D, Dropout, MaxPooling1D


class NeuralNetwork:
    def __init__(self, model_type, x, y, max_features=5000, max_input_length=500):
        self.input = x
        self.response = y
        self.batch_size = 32
        self.epochs = 15
        self.max_features = max_features
        self.max_input_length = max_input_length
        if model_type == 'rnn1':
            self.network = self.rnn1()
        elif model_type == 'rnn2':
            self.network = self.rnn2()
        else:
            self.network = self.ffn()

    def ffn(self):
        input_dim = self.input.shape[1]  # Number of features
        model = Sequential()
        model.add(Dense(64, input_dim=input_dim, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
        model.summary()
        return model

    def rnn1(self):
        model = Sequential()
        model.add(Embedding(self.max_features, 32, input_length=self.max_input_length))
        model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model

    def rnn2(self):
        model = Sequential()
        model.add(Embedding(self.max_features, 128))
        model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model

    def train(self, x_train, y_train, x_test, y_test, batch_size=32, epochs=15, verbose=True):
        self.network.fit(x_train, y_train, epochs=epochs, verbose=verbose, batch_size=batch_size)
        (loss, accuracy) = self.network.evaluate(x_train, y_train)
        print("Training Accuracy: {:.4f}".format(accuracy))
        (loss, accuracy) = self.network.evaluate(x_test, y_test)
        print("Testing Accuracy:  {:.4f}".format(accuracy))
