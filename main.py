import datetime
import ntmTest
import keras
import numpy as np
from keras.utils import plot_model
from copyTask import get_sample

LOG_PATH_BASE="./logs/"
output_dim = 8
input_dim = output_dim + 2  # this is the actual input dim of the network, that includes two dims for flags
batch_size = 100
testrange=[5,10,20,40,80,160]
epochs=1800



def test_model(model, file_name=None, min_size=40, batch_size=128, input_dim=10):
    I, V, sw = get_sample(batch_size=batch_size, in_bits=input_dim, out_bits=output_dim, max_size=min_size, min_size=min_size)
    Y = np.asarray(model.predict(I, batch_size=batch_size)).astype('float64')
    ntm = model.layers[0]
    k, b, M = ntm.get_weights()
    if not np.isnan(Y.sum()): #checks for a NaN anywhere
        Y = (Y > 0.5).astype('float64')
        #print(Y)
        acc = (V[:, -min_size:, :] == Y[:, -min_size:, :]).mean() * 100
    else:
        ntm = model.layers[0]
        k, b, M = ntm.get_weights()
        import pudb; pu.db
        acc = 0
    return acc

def train_model(model, batch_size=128, epochs=10, validation_split=0, min_size=5, max_size=20, callbacks=None, input_dim=10):

    for i in range(epochs): 
        I, V, sw = get_sample(batch_size=1000, in_bits=input_dim, out_bits=output_dim, max_size=max_size, min_size=min_size)
        model.fit(I, V, sample_weight=sw, callbacks=callbacks, epochs=i+1, batch_size=batch_size, initial_epoch=i)

        ntm = model.layers[0]
        k, b, M = ntm.get_weights()

    print("done training")


def lengthy_test(model, batch_size=100, input_dim=10, output_dim=8):
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    log_path = LOG_PATH_BASE + ts + "_-_" + model.name 
    tb = keras.callbacks.TensorBoard(log_dir=log_path, write_graph=True)
    
    callbacks = [tb, keras.callbacks.TerminateOnNaN()]

    for i in testrange:
        acc = test_model(model, min_size=i, batch_size=batch_size, input_dim=input_dim)
        print("the accuracy for length {0} was: {1}%%".format(i,acc))
    train_model(model, epochs=epochs, batch_size=batch_size, callbacks=callbacks, input_dim=input_dim)
    for i in testrange:
        import pudb; pu.db
        acc = test_model(model, min_size=i, batch_size=batch_size, input_dim=input_dim)
        print("the accuracy for length {0} was: {1}%%".format(i,acc))
    return


model_NTM, batch_size = ntmTest.gen_model(input_dim=input_dim, output_dim=output_dim, batch_size=batch_size)

lengthy_test(model_NTM, batch_size=batch_size, input_dim=input_dim, output_dim=output_dim) #, testrange = [5,10,20])
import pudb; pu.db

