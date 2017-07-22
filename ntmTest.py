import numpy as np

from keras.layers.core import Activation
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.optimizers import Adam
from keras import backend as K
import keras

from ntm import NeuralTuringMachine as NTM


n_slots = 128
m_length = 20
learning_rate = 5e-4
clipnorm = 10

def gen_model(input_dim, batch_size, output_dim):
    model = Sequential()
    model.name = "NTM"
    ntm = NTM(output_dim, n_slots=n_slots, m_length=m_length, shift_range=3,
              controller_architecture='dense',
              return_sequences=True,
              input_shape=(None, input_dim), 
              batch_size = batch_size)
    model.add(ntm)
    #model.add(Activation('sigmoid'))

    sgd = Adam(lr=learning_rate, clipnorm=clipnorm)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics = ['binary_accuracy'], sample_weight_mode="temporal")

    return model, batch_size


