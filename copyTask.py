import numpy as np
import datetime
import keras

LOG_PATH_BASE="./logs/"



def get_sample(batch_size=128, in_bits=10, out_bits=8, max_size=20, min_size=1):
    # generate samples with random length.
    # there a two flags, one for the beginning of the sequence 
    # (only second to last bit is one)
    # and one for the end of the sequence (only last bit is one)
    # every other time those are always zero.
    # therefore the length of the generated sample is:
    # 1 + actual_sequence_length + 1 + actual_sequence_length
    
    # make flags
    begin_flag = np.zeros((1, in_bits))
    begin_flag[0, in_bits-2] = 1
    end_flag = np.zeros((1, in_bits))
    end_flag[0, in_bits-1] = 1

    # initialize arrays: for processing, every sequence must be of the same length.
    # We pad with zeros.
    temporal_length = max_size*2 + 2
    # "Nothing" on our band is represented by 0.5 to prevent immense bias towards 0 or 1.
    inp = np.ones((batch_size, temporal_length, in_bits))*0.5
    out = np.ones((batch_size, temporal_length, out_bits))*0.5
    # sample weights: in order to make recalling the sequence much more important than having everything set to 0
    # before and after, we construct a weights vector with 1 where the sequence should be recalled, and small values
    # anywhere else.
    sw  = np.ones((batch_size, temporal_length))*0.01

    # make actual sequence
    for i in range(batch_size):
        ts = np.random.randint(low=min_size, high=max_size+1)
        actual_sequence = np.random.uniform(size=(ts, out_bits)) > 0.5
        output_sequence = np.concatenate((np.ones((ts+2, out_bits))*0.5, actual_sequence), axis=0)

        # pad with zeros where only the flags should be one
        padded_sequence = np.concatenate((actual_sequence, np.zeros((ts, 2))), axis=1)
        input_sequence = np.concatenate((begin_flag, padded_sequence, end_flag), axis=0)
        

        # this embedds them, padding with the neutral value 0.5 automatically
        inp[i, :input_sequence.shape[0]] = input_sequence
        out[i, :output_sequence.shape[0]] = output_sequence
        sw[i, ts+2 : ts+2+ts] = 1

    return inp, out, sw

def test_model(model, file_name=None, min_size=40, batch_size=128, input_dim=10):
    I, V, sw = get_sample(batch_size=batch_size, n_bits=input_dim, max_size=min_size, min_size=min_size)
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
        I, V, sw = get_sample(batch_size=1000, n_bits=input_dim, max_size=max_size, min_size=min_size)
        model.fit(I, V, sample_weight=sw, callbacks=callbacks, epochs=i+1, batch_size=batch_size, initial_epoch=i)

        ntm = model.layers[0]
        k, b, M = ntm.get_weights()

    print("done training")


def lengthy_test(model, testrange=[5,10,20,40,80], training_epochs=512, batch_size=100, input_dim=10):
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    log_path = LOG_PATH_BASE + ts + "_-_" + model.name 
    tb = keras.callbacks.TensorBoard(log_dir=log_path, write_graph=True)
    
    callbacks = [tb, keras.callbacks.TerminateOnNaN()]

    for i in testrange:
        acc = test_model(model, min_size=i, batch_size=batch_size, input_dim=input_dim)
        print("the accuracy for length {0} was: {1}%%".format(i,acc))
    train_model(model, epochs=training_epochs, batch_size=batch_size, callbacks=callbacks, input_dim=input_dim)
    for i in testrange:
        import pudb; pu.db
        acc = test_model(model, min_size=i, batch_size=batch_size, input_dim=input_dim)
        print("the accuracy for length {0} was: {1}%%".format(i,acc))
    return

