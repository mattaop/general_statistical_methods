from keras.models import Model
from keras.layers import Input, Dense, Embedding, LSTM,  Dropout, Flatten, concatenate, BatchNormalization
import matplotlib.pyplot as plt
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint


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
        self.max_features_img = max_features[11]
        self.network = self._neural_network()

    def _root_mean_squared_error(self, y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

    def _neural_network(self):
        #################
        # Define inputs #
        #################
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
        img = Input(shape=(1,))

        #############
        # Text data #
        #############
        embedding_layer = Embedding(input_dim=self.max_features_text, output_dim=32)

        def encoder(input_data):
            x = embedding_layer(input_data)
            x = LSTM(32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(x)
            x = LSTM(32, dropout=0.2, recurrent_dropout=0.2)(x)
            x = Dense(32, activation='relu')(x)
            #x = Dense(1, activation='sigmoid')(x)
            return BatchNormalization()(x)

        """
        x1 = embedding_layer(title)
        x1 = LSTM(32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(x1)
        x1 = LSTM(32, dropout=0.2, recurrent_dropout=0.2)(x1)
        x1 = Dense(32, activation='relu')(x1)
        x1 = Dense(1, activation='sigmoid')(x1)
        x1 = BatchNormalization()(x1)
        """

        x1 = encoder(title)
        x2 = encoder(desc)

        """
        x2 = embedding_layer(desc)
        x2 = LSTM(32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(x2)
        x2 = LSTM(32, dropout=0.2, recurrent_dropout=0.2)(x2)
        x2 = Dense(32, activation='relu')(x2)
        x2 = Dense(1, activation='sigmoid')(x2)
        x2 = BatchNormalization()(x2)
        """

        ####################
        # Categorical data #
        ####################
        def dense_layers(input_data, units):
            y = Flatten()(input_data)
            y = Dense(units, activation='relu')(y)
            y = Dense(4, activation='relu')(y)
            return BatchNormalization()(y)

        y1 = dense_layers(Embedding(input_dim=self.max_features_region, output_dim=16)(region), 16)  # Layers for region
        y2 = dense_layers(Embedding(input_dim=self.max_features_city, output_dim=16)(city), 16)  # Layers for city
        y3 = dense_layers(Embedding(input_dim=self.max_features_parent_category_name, output_dim=16)(cat1), 16)  # Layers for cat1
        y4 = dense_layers(Embedding(input_dim=self.max_features_category_name, output_dim=16)(cat2), 16)  # Layers for cat2
        y5 = dense_layers(Embedding(input_dim=self.max_features_param1, output_dim=16)(param1), 16)  # Layers for param1
        y6 = dense_layers(Embedding(input_dim=self.max_features_param2, output_dim=16)(param2), 16)  # Layers for param2
        y7 = dense_layers(Embedding(input_dim=self.max_features_param3, output_dim=16)(param3), 16)  # Layers for param3
        y8 = dense_layers(Embedding(input_dim=self.max_features_user_type, output_dim=16)(user_type), 16)  # Layers for user_type
        y9 = dense_layers(Embedding(input_dim=self.max_features_date, output_dim=16)(date), 16)  # Layers for date
        y10 = Dense(16, activation='relu')(item_number)  # Layers for item type
        y10 = Dense(16, activation='relu')(y10)  # Layers for item type
        y10 = BatchNormalization()(y10)
        y11 = Dense(16, activation='relu')(price)  # Layers for price
        y11 = Dense(16, activation='relu')(y11)  # Layers for price
        y11 = BatchNormalization()(y11)

        y12 = dense_layers(Embedding(input_dim=self.max_features_img, output_dim=16)(img), 16)

        ####################
        # Combine networks #
        ####################
        z = concatenate([x1, x2, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12])
        z = Dense(128, activation="relu")(z)
        z = BatchNormalization()(z)
        z = Dropout(0.2)(z)
        z = Dense(64, activation="relu")(z)
        z = Dense(1, activation="sigmoid")(z)

        ################
        # Define model #
        ################
        model = Model(inputs=[title, desc, region, city, cat1, cat2, date, param1, param2, param3, user_type,
                              item_number, price, img], outputs=z)
        model.compile(loss=self._root_mean_squared_error, optimizer='Rmsprop', metrics=['accuracy'])
        model.summary()

        return model

    def train(self, data_train, data_test, batch_size=64, epochs=50, verbose=True):
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
        save_weights = ModelCheckpoint('weights.best.hdf5')
        callbacks = [early_stopping, save_weights]
        history = self.network.fit(data_train.x, data_train.y, epochs=epochs, verbose=verbose, batch_size=batch_size,
                                   callbacks=callbacks, validation_data=[data_test.x, data_test.y])
        self._plot_results(history)

    def test(self, data):
        history = self.network.evaluate(data.x, data.y)
        print("Test loss: ", history[0], ", test accuracy: ", history[1])
        print("Predicting values...")
        predictions = self.network.predict(input)
        self._plot_prediction(predictions, data.y)

    def save_weight(self):
        # serialize model to JSON
        model_json = self.network.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.network.save_weights("model.h5")
        print("Saved model to disk")

    def _plot_results(self, history):
        # show the terrible predictions
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

    def _plot_prediction(self, predictions, label):
        plt.scatter(predictions, label)
        plt.show()

