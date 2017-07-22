import keras
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import Adam

lr = 5e-4
clipnorm = 10
units = 256



def gen_model(input_dim=10, output_dim=8, batch_size=100):
    model_LSTM = Sequential()
    model_LSTM.name = "FFW"
    model_LSTM.batch_size = batch_size
    model_LSTM.input_dim = input_dim
    model_LSTM.output_dim = output_dim

    model_LSTM.add(Dense(input_shape=(None, input_dim), units=output_dim))
    model_LSTM.add(Activation('sigmoid'))

    sgd = Adam(lr=lr, clipnorm=clipnorm)
    model_LSTM.compile(loss='binary_crossentropy', optimizer=sgd, metrics = ['binary_accuracy'], sample_weight_mode="temporal")

    return model_LSTM
