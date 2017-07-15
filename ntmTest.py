
from __future__ import absolute_import
from __future__ import print_function
import logging
import numpy as np
np.random.seed(124)
import matplotlib.pyplot as plt
#import cPickle

from theano import tensor, function

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dropout, Activation, Flatten, Masking, Dense
from keras.layers.recurrent import LSTM
from keras.utils import np_utils, generic_utils
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam, SGD
from keras import backend as KB

from ntm import NeuralTuringMachine as NTM

from IPython import display

batch_size = 100

h_dim = 64
n_slots = 50
m_length = 20
input_dim = 8
lr = 1e-3
clipnorm = 10

# Neural Turing Machine


def gen_model():
    KB.clear_session()
    model = Sequential()
    ntm = NTM(2*h_dim, n_slots=n_slots, m_length=m_length, shift_range=3,
              inner_rnn='lstm',
              return_sequences=True,
              input_shape=(None, input_dim), 
              batch_size = batch_size
              )
    model.add(ntm)
    import pudb; pu.db
    model.add(TimeDistributed(Dense(units=input_dim)))
    model.add(Activation('sigmoid'))

    sgd = Adam(lr=lr, clipnorm=clipnorm)
    model.compile(loss='binary_crossentropy', optimizer=sgd, sample_weight_mode="temporal")
