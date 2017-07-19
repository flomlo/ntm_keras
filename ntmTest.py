
import logging
import numpy as np

from copyTask import lengthy_test

from keras.layers.core import Activation
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras import backend as K

from ntm import NeuralTuringMachine as NTM


batch_size = 1

output_dim = 8
n_slots = 50
m_length = 20
input_dim = 8
lr = 1e-3
clipnorm = 10

def gen_model():
    model = Sequential()
    model.name = "NTM"
    ntm = NTM(output_dim, n_slots=n_slots, m_length=m_length, shift_range=3,
              controller_architecture='dense',
              return_sequences=True,
              input_shape=(None, input_dim), 
              batch_size = batch_size)
    model.add(ntm)
    model.add(Activation('linear'))

    sgd = Adam(lr=lr) #, clipnorm=clipnorm)
    model.compile(loss='mean_absolute_error', optimizer=sgd, metrics = ['accuracy'], sample_weight_mode="temporal")
#    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics = ['accuracy'], sample_weight_mode="temporal")

    return model, batch_size


