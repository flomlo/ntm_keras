import numpy as np
import matplotlib.pyplot as plt
import datetime
import pathlib
import keras

input_dim = 8


LOG_PATH_BASE="./logs/"




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


def test_model(model, file_name=None, min_size=40, batch_size=128):
    I, V, sw = get_sample(batch_size=batch_size, n_bits=input_dim, max_size=min_size+1, min_size=min_size)
    #import pudb; pu.db
    Y = np.asarray(model.predict(I, batch_size=batch_size) > .5).astype('float64')
    acc = (V[:, -min_size:, :] == Y[:, -min_size:, :]).mean() * 100
    #show_pattern(Y[0], V[0], sw[0], file_name)
    return acc

def train_model(model, batch_size=128, epochs=10, validation_split=0, min_size=5, max_size=20, callbacks=None):
    I, V, sw = get_sample(batch_size=batch_size, n_bits=input_dim, max_size=min_size+1, min_size=min_size)
    import pudb; pu.db
    model.fit(I, V, callbacks=callbacks, epochs=epochs, batch_size=batch_size, validation_split=validation_split)


def lengthy_test(model, testrange=[5,10,20,40,80], training_epochs=512, batch_size=100):
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    log_path = LOG_PATH_BASE + ts + "_-_" + model.name 
    tb = keras.callbacks.TensorBoard(log_dir=log_path, write_graph=True)
    
    callbacks = [tb, keras.callbacks.TerminateOnNaN()]
    for i in testrange:
        acc = test_model(model, min_size=i, batch_size=batch_size)
        print("the accuracy for length {0} was: {1}%%".format(i,acc))
    print("testing without training done. now training")
    train_model(model, epochs=training_epochs, batch_size=batch_size, callbacks=callbacks)
    print("training complete, now testing again") 
    for i in testrange:
        acc = test_model(model, min_size=i, batch_size=batch_size)
        print("the accuracy for length {0} was: {1}%%".format(i,acc))
    return

