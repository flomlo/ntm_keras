import numpy as np
import matplotlib.pyplot as plt
import datetime
import pathlib

import keras
from keras.models import Sequential
from keras.layers import LSTM, Activation, Dense
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import Adam

LOG_PATH_BASE="./logs/"

batch_size = 128

h_dim = 64
n_slots = 50
m_length = 20
input_dim = 8
lr = 1e-3
clipnorm = 10

input_shape = (None, input_dim)

sgd = Adam(lr=lr, clipnorm=clipnorm)


model_LSTM = Sequential()
model_LSTM.add(LSTM(input_shape=input_shape, units=h_dim*2, return_sequences=True))
model_LSTM.add(LSTM(units=h_dim*2, return_sequences=True))
model_LSTM.add(LSTM(units=h_dim*2, return_sequences=True))
model_LSTM.add(TimeDistributed(Dense(input_dim)))
model_LSTM.add(Activation('sigmoid'))

model_LSTM.compile(loss='binary_crossentropy', optimizer=sgd)



def get_sample(batch_size=128, n_bits=8, max_size=20, min_size=1):
    # generate samples with random length
    inp = np.zeros((batch_size, 2*max_size-1, n_bits))
    out = np.zeros((batch_size, 2*max_size-1, n_bits))
    sw = np.zeros((batch_size, 2*max_size-1, 1))
    for i in range(batch_size):
        t = np.random.randint(low=min_size, high=max_size)
        x = np.random.uniform(size=(t, n_bits)) > .5
        for j,f in enumerate(x.sum(axis=-1)): # remove fake flags
            if f>=n_bits:
                x[j, :] = 0.
        del_flag = np.ones((1, n_bits))
        inp[i, :t+1] = np.concatenate([x, del_flag], axis=0)
        out[i, t:(2*t)] = x
        sw[i, t:(2*t)] = 1
    return inp, out, sw

def show_pattern(inp, out, sw, file_name='pattern2.png'):
    plt.figure(figsize=(10, 10))
    plt.subplot(131)
    #plt.imshow(inp>.5)
    plt.subplot(132)
    #plt.imshow(out>.5)
    plt.subplot(133)
    #plt.imshow(sw[:, :1]>.5)
    plt.savefig(file_name)
    plt.close()


def test_model(model, file_name=None, min_size=40):
    I, V, sw = get_sample(batch_size=500, n_bits=input_dim, max_size=min_size+1, min_size=min_size)
    Y = np.asarray(model.predict(I, batch_size=100) > .5).astype('float64')
    acc = (V[:, -min_size:, :] == Y[:, -min_size:, :]).mean() * 100
    #show_pattern(Y[0], V[0], sw[0], file_name)
    return acc

def train_model(model, batch_size=128, epochs=10, validation_split=0, min_size=5, max_size=20, logger=None):
    I, V, sw = get_sample(batch_size=batch_size, n_bits=input_dim, max_size=min_size+1, min_size=min_size)
    model.fit(I, V, callbacks=[logger], epochs=epochs, batch_size=batch_size, validation_split=validation_split)


def lengthy_test(model, testrange=[5,10,20,40,80], training_epochs=512):
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    log_path = LOG_PATH_BASE + ts + "_-_" + "LSTM" 
    print ("creating:" + log_path) 
    #pathlib.Path(log_path).mkdir(parents=True, exist_ok=False)
    tb = keras.callbacks.TensorBoard(log_dir=log_path, write_graph=True)
    for i in testrange:
        acc = test_model(model, min_size=i)
        print("the accuracy for length {0} was: {1}%%".format(i,acc))
    print("testing without training done. now training")
    train_model(model_LSTM, epochs=training_epochs, batch_size=128, logger=tb)
    print("training complete, now testing again") 
    for i in testrange:
        acc = test_model(model, min_size=i)
        print("the accuracy for length {0} was: {1}%%".format(i,acc))
    tb = None
    return

