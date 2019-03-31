from keras.models import Model
from keras.layers import Input, Dense, Embedding, LSTM,  Dropout, Flatten, concatenate, BatchNormalization, \
    GlobalMaxPooling1D, Conv1D, MaxPooling1D
from keras.initializers import RandomNormal
import matplotlib.pyplot as plt
import keras.backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.optimizers import adam


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

    def gauss_init(self):
        return RandomNormal(mean=0.0, stddev=0.005)

    def schedule(self, epoch):
        return 0.01*10**(-epoch)

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
        embedding_layer = Embedding(input_dim=self.max_features_text, output_dim=64,
                                    embeddings_initializer=self.gauss_init())

        def encoderLSTM(input_data):
            x = embedding_layer(input_data)
            x = LSTM(64, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(x)
            x = LSTM(64, dropout=0.2, recurrent_dropout=0.2)(x)
            # x = Dense(16, activation='relu')(x)
            # x = Dropout(0.2)(x)
            x = GlobalMaxPooling1D()(x)
            # x = Dense(1, activation='softmax')(x)
            return x

        def encoderCNN(input_data):
            x = embedding_layer(input_data)
            x = Conv1D(64, 7, activation='relu', padding='same')(x)
            x = MaxPooling1D(2)(x)
            x = Conv1D(64, 7, activation='relu', padding='same')(x)
            x = GlobalMaxPooling1D()(x)
            return x

        # x1 = encoderLSTM(title)
        # x2 = encoderLSTM(desc)
        # x1 = encoderCNN(title)
        # x2 = encoderCNN(desc)
        x1 = embedding_layer(title)
        x1 = GlobalMaxPooling1D()(x1)
        x2 = embedding_layer(desc)
        x2 = GlobalMaxPooling1D()(x2)

        ####################
        # Categorical data #
        ####################
        embedding_dim = 80

        def dense_layers(input_data, units=embedding_dim):
            y = Flatten()(input_data)
            y = Dense(units, activation='relu')(y)
            y = Dropout(0.4)(y)
            # y = Dense(16, activation='relu')(y)
            # y = Dropout(0.2)(y)
            # y = Dense(1, activation='softmax')(y)
            return y

        y1 = dense_layers(Embedding(input_dim=self.max_features_region, output_dim=embedding_dim,
                                    embeddings_initializer=self.gauss_init())(region))  # Layers for region
        y2 = dense_layers(Embedding(input_dim=self.max_features_city, output_dim=embedding_dim,
                                    embeddings_initializer=self.gauss_init())(city))  # Layers for city
        y3 = dense_layers(Embedding(input_dim=self.max_features_parent_category_name, output_dim=embedding_dim,
                                    embeddings_initializer=self.gauss_init())(cat1))  # Layers for cat1
        y4 = dense_layers(Embedding(input_dim=self.max_features_category_name, output_dim=embedding_dim,
                                    embeddings_initializer=self.gauss_init())(cat2))  # Layers for cat2
        y5 = dense_layers(Embedding(input_dim=self.max_features_param1, output_dim=embedding_dim,
                                    embeddings_initializer=self.gauss_init())(param1))  # Layers for param1
        y6 = dense_layers(Embedding(input_dim=self.max_features_param2, output_dim=embedding_dim,
                                    embeddings_initializer=self.gauss_init())(param2))  # Layers for param2
        y7 = dense_layers(Embedding(input_dim=self.max_features_param3, output_dim=embedding_dim,
                                    embeddings_initializer=self.gauss_init())(param3))  # Layers for param3
        y8 = dense_layers(Embedding(input_dim=self.max_features_user_type, output_dim=embedding_dim,
                                    embeddings_initializer=self.gauss_init())(user_type))  # Layers for user_type
        y9 = dense_layers(Embedding(input_dim=self.max_features_date, output_dim=embedding_dim,
                                    embeddings_initializer=self.gauss_init())(date))  # Layers for date
        y10 = dense_layers(Embedding(input_dim=self.max_features_img, output_dim=embedding_dim,
                                     embeddings_initializer=self.gauss_init())(img))  # Layers for image type

        y11 = Dense(16, activation='relu')(item_number)  # Layers for item type
        y11 = Dropout(0.4)(y11)
        y12 = Dense(16, activation='relu')(price)  # Layers for price
        y12 = Dropout(0.4)(y12)

        ####################
        # Combine networks #
        ####################
        z = concatenate([x1, x2, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12])
        z = BatchNormalization()(z)
        z = Dense(512, activation="relu")(z)
        z = Dropout(0.4)(z)
        z = Dense(64, activation="relu")(z)
        z = Dropout(0.4)(z)
        z = Dense(1, activation="sigmoid")(z)

        # Optimizer
        opt = adam(lr=0.001, decay=1e-6)

        ################
        # Define model #
        ################
        model = Model(inputs=[title, desc, region, city, cat1, cat2, date, param1, param2, param3, user_type,
                              item_number, price, img], outputs=z)
        model.compile(loss=self._root_mean_squared_error, optimizer=opt, metrics=['accuracy'])
        model.summary()

        return model

    def train(self, data_train, batch_size=64, epochs=50, verbose=True):
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
        save_weights = ModelCheckpoint('weights.best.h5')
        set_learning_rate = LearningRateScheduler(schedule=self.schedule)
        callbacks = [early_stopping, save_weights]
        history = self.network.fit(data_train.x, data_train.y, epochs=epochs, verbose=verbose, batch_size=batch_size,
                                   callbacks=callbacks, validation_split=0.2)
        self._plot_results(history)

    def test(self, data):
        history = self.network.evaluate(data.x, data.y)
        print("Test loss: ", history[0], ", test accuracy: ", history[1])
        print("Predicting values...")
        predictions = self.network.predict(data.x)
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
        plt.scatter(predictions, label, s=1)
        plt.title('Predicted vs true values')
        plt.ylabel('Deal probability')
        plt.xlabel('Predicted deal probability')
        plt.show()

