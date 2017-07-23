import keras
from keras.layers.core import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam

from model_ntm import gen_model
from testing_utils import lengthy_test

lr = 3e-5
clipnorm = 10
sgd = Adam(lr=lr, clipnorm=clipnorm)


controller = Sequential()
controller.name="DD-FFW"
controller.add(Dense(units=300,
                        #kernel_initializer='random_normal', 
                        bias_initializer='random_normal',
                        activation='sigmoid',
                        input_dim=30))
#controller.add(Dropout(0.5))
controller.add(Dense(units=100,
                        #kernel_initializer='random_normal', 
                        bias_initializer='random_normal',
                        activation='sigmoid'))
controller.compile(loss='binary_crossentropy', optimizer=sgd, metrics = ['binary_accuracy'], sample_weight_mode="temporal")

model = gen_model(input_dim=10, output_dim=8, batch_size=100,
        controller_model=controller)

lengthy_test(model, epochs=1000)

import pudb, pu.db
