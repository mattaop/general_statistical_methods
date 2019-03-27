from keras.models import Sequential,Input,Model
from keras.layers import Input, Dense, Embedding, SimpleRNN, LSTM, Conv1D, Dropout, MaxPooling1D, Flatten, concatenate, BatchNormalization
import matplotlib.pyplot as plt
import keras.backend as K
import numpy as np


class NeuralNetwork:
    def __init__(self, max_title, max_desc, max_features):
        self.max_title = max_title
        self.max_desc = max_desc
        self.max_features_text = max_features[0]
        self.max_features_region = max_features[2]
        self.max_features_city = max_features[3]
        self.max_features_parent_category_name = max_features[4]
        self.max_features_category_name = max_features[5]
        self.max_features_param1 = max_features[6]
        self.max_features_param2 = max_features[7]
        self.max_features_param3 = max_features[8]
        self.max_features_user_type = max_features[9]
        self.max_features_date = max_features[10]
        self.network = self.neural_network()

    def root_mean_squared_error(self, y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

    def neural_network(self):
        # define two sets of inputs
        title = Input(shape=(self.max_title,))
        desc = Input(shape=(self.max_desc,))
        region = Input(shape=(5,))
        city = Input(shape=(5,))
        cat1 = Input(shape=(10,))
        cat2 = Input(shape=(10,))
        param1 = Input(shape=(5,))
        param2 = Input(shape=(5,))
        param3 = Input(shape=(5,))
        user_type = Input(shape=(3,))
        item_number = Input(shape=(1,))
        date = Input(shape=(1,))
        price = Input(shape=(1,))

        # the first network operating on title
        embedding_layer = Embedding(input_dim=self.max_features_text, output_dim=32)

        x1 = embedding_layer(title)
        x1 = LSTM(32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(x1)
        x1 = LSTM(32, dropout=0.2, recurrent_dropout=0.2)(x1)
        #x1 = Flatten()(x1)
        x1 = Dense(32, activation='relu')(x1)
        x1 = Dense(1, activation='sigmoid')(x1)

        x2 = embedding_layer(desc)
        x2 = LSTM(32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(x2)
        x2 = LSTM(32, dropout=0.2, recurrent_dropout=0.2)(x2)
        #x2 = Flatten()(x2)
        x2 = Dense(32, activation='relu')(x2)
        x2 = Dense(1, activation='sigmoid')(x2)

        # the third layer containing categorical values (region, city, category and price)
        y1 = Embedding(input_dim=self.max_features_region, output_dim=4)(region)
        y1 = Flatten()(y1)
        y1 = Dense(4, activation='relu')(y1)

        y2 = Embedding(input_dim=self.max_features_city, output_dim=4)(city)
        y2 = Flatten()(y2)
        y2 = Dense(4, activation='relu')(y2)

        y3 = Embedding(input_dim=self.max_features_parent_category_name, output_dim=4)(cat1)
        y3 = Flatten()(y3)
        y3 = Dense(4, activation='relu')(y3)

        y4 = Embedding(input_dim=self.max_features_category_name, output_dim=4)(cat2)
        y4 = Flatten()(y4)
        y4 = Dense(4, activation='relu')(y4)

        y5 = Embedding(input_dim=self.max_features_param1, output_dim=4)(param1)
        y5 = Flatten()(y5)
        y5 = Dense(4, activation='relu')(y5)

        y6 = Embedding(input_dim=self.max_features_param2, output_dim=4)(param2)
        y6 = Flatten()(y6)
        y6 = Dense(4, activation='relu')(y6)

        y7 = Embedding(input_dim=self.max_features_param3, output_dim=4)(param3)
        y7 = Flatten()(y7)
        y7 = Dense(4, activation='relu')(y7)

        y8 = Embedding(input_dim=self.max_features_user_type, output_dim=4)(user_type)
        y8 = Flatten()(y8)
        y8 = Dense(4, activation='relu')(y8)

        y9 = Embedding(input_dim=self.max_features_date, output_dim=4)(date)
        y9 = Flatten()(y9)
        y9 = Dense(4, activation='relu')(y9)

        y10 = Dense(4, activation='relu')(item_number)

        y11 = Dense(4, activation='relu')(price)

        # combine the outputs
        z = concatenate([x1, x2, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11])
        #z = BatchNormalization()(z)
        z = Dense(13, activation="relu")(z)
        z = Dropout(0.2)(z)
        z = Dense(13, activation="relu")(z)
        z = Dense(1, activation="sigmoid")(z)

        # define inputs and outputs
        model = Model(inputs=[title, desc, region, city, cat1, cat2, date, param1, param2, param3, user_type,
                              item_number, price], outputs=z)

        model.compile(loss=self.root_mean_squared_error, optimizer='Rmsprop', metrics=['accuracy'])
        model.summary()

        return model

    def train(self, data, batch_size=32, epochs=1, verbose=True):
        input = [data.title, data.desc, data.region, data.city, data.cat1, data.cat2, data.date,
                 data.param1, data.param2, data.param3, data.user_type, data.item_number, data.price]
        history = self.network.fit(input, data.y, epochs=epochs, verbose=verbose, batch_size=batch_size, validation_split=0.2)
        self.plot_results(history)

    def test(self, data):
        input = [data.title, data.desc, data.region, data.city, data.cat1, data.cat2, data.date,
                 data.param1, data.param2, data.param3, data.user_type, data.item_number, data.price]
        history = self.network.evaluate(input, data.y)
        print("Test loss: ", history[0], ", test accuracy: ", history[1])
        print("Predicting values...")
        predictions = self.network.predict(input)
        self.plot_prediction(predictions, data.y)

    def save_weight(self):
        # serialize model to JSON
        model_json = self.network.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.network.save_weights("model.h5")
        print("Saved model to disk")

    def plot_results(self, history):
        # show the terrible predictions
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

    def plot_prediction(self, predictions, label):
        plt.scatter(predictions, label)
        plt.show()
