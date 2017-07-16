import keras
from keras.models import Sequential
from keras.layers import LSTM, Activation, Dense
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam

batch_size = 128
h_dim = 64
n_slots = 50
m_length = 20
input_dim = 8
lr = 1e-3
clipnorm = 10

input_shape = (None, input_dim)


def gen_model():
    model_LSTM = Sequential()
    model_LSTM.name = "LSTM"
    model_LSTM.add(LSTM(input_shape=input_shape, units=h_dim*2, return_sequences=True))
    model_LSTM.add(LSTM(units=h_dim*2, return_sequences=True))
    model_LSTM.add(LSTM(units=h_dim*2, return_sequences=True))
    model_LSTM.add(TimeDistributed(Dense(input_dim)))
    model_LSTM.add(Activation('sigmoid'))


    sgd = Adam(lr=lr, clipnorm=clipnorm)
    model_LSTM.compile(loss='binary_crossentropy', optimizer=sgd)

    return model_LSTM
