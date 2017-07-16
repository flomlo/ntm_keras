import numpy as np

import theano
import theano.tensor as T
floatX = theano.config.floatX

from keras.layers.recurrent import Recurrent, GRU, LSTM
from keras.initializers import Orthogonal, Zeros
from keras import backend as K
from keras.engine.topology import InputSpec 

from utils import rnn_states




tol = 1e-4


def _wta(X):
    M = K.max(X, axis=-1, keepdims=True)
    R = K.switch(K.equal(X, M), X, 0.)
    return R


def _update_controller(self, inp, h_tm1, M):
    """We have to update the inner RNN inside the NTM, this
    is the function to do it. Pretty much copy+pasta from Keras
    """
    x = K.concatenate([inp, M], axis=-1)
    #1 is for gru, 2 is for lstm
    if len(h_tm1) in [1,2]:
        if hasattr(self.rnn,"get_constants"):
            BW,BU = self.rnn.get_constants(x)
            h_tm1 += (BW,BU)
    # update state
    _, h = self.rnn.step(self.rnn.preprocess_input(x), h_tm1)
#   this is one step in the right direction should the LSTM work with
#   implementation=0; but so far doesnt work (BUG?)
#    _, h = self.rnn.step(self.rnn.preprocess_input(K.repeat(x,1)), h_tm1)

#    import pudb; pu.db

    return h


def _circulant(leng, n_shifts):
    """
    I confess, I'm actually proud of this hack. I hope you enjoy!
    This will generate a tensor with `n_shifts` of rotated versions the
    identity matrix. When this tensor is multiplied by a vector
    the result are `n_shifts` shifted versions of that vector. Since
    everything is done with inner products, everything is differentiable.

    Paramters:
    ----------
    leng: int > 0, number of memory locations
    n_shifts: int > 0, number of allowed shifts (if 1, no shift)

    Returns:
    --------
    shift operation, a tensor with dimensions (n_shifts, leng, leng)
    """
    eye = np.eye(leng)
    shifts = range(n_shifts//2, -n_shifts//2, -1)
    C = np.asarray([np.roll(eye, s, axis=1) for s in shifts])
    return K.variable(C.astype(K.floatx()))


def _renorm(x):
    return x / (K.sum(x, axis=1, keepdims=True))


def _softmax(x):
    # that is probably not even useful anymore. what did it do?
    wt = K.batch_flatten(x)
    w = K.softmax(wt)
    return w.reshape(x.shape)  # T.clip(s, 0, 1)


def _cosine_distance(M, k):
    # this is equation (6)
    # TODO: Is this the best way to implement it? 
    # Can it be found in a library?
    # TODO: probably besser conditioned if we first normalize and then do the scalar product
    dot = K.sum((M * k[:, None, :]),axis=-1)
    nM = K.sqrt(K.sum((M**2)))
    nk = K.sqrt(K.sum((k**2), axis=-1, keepdims=True))
    return dot / (nM * nk)


class NeuralTuringMachine(Recurrent):
    """ Neural Turing Machines

    Non obvious parameter:
    ----------------------
    shift_range: int, number of available shifts, ex. if 3, avilable shifts are
                 (-1, 0, 1)
    n_slots: number of memory locations, defined in 3.1 as N
    m_length: memory length at each location, defined in 3.1 as M

    Known issues:
    -------------
    Theano may complain when n_slots == 1.
    Currently batch_input_size is necessary. Or not? Im not even sure :(

    """
    def __init__(self, output_dim, n_slots, m_length, shift_range=3,
                        inner_rnn='lstm',
                        batch_size=777,                 
#                       init='glorot_uniform',
#                       inner_init='orthogonal',      Default values anyway
                        **kwargs):
        super(NeuralTuringMachine, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.units = output_dim
        self.n_slots = n_slots
        self.m_length = m_length
        self.shift_range = shift_range
        self.inner_rnn = inner_rnn
        self.batch_size = batch_size
        
        # WARNING: Not understood, only copied from keras/recurrent.py
        # In our case the dimension seems to be 5 (LSTM) or 4 (GRU),
        # see get_initial_states
        self.state_spec = [InputSpec(shape=(None, self.units)),
                            InputSpec(shape=(None, self.units)),
                            InputSpec(shape=(None, self.units)),
                            InputSpec(shape=(None, self.units)),
                            InputSpec(shape=(None, self.units))]


    def build(self, input_shape):
        print("here comes the input shape")
        print(input_shape)
        bs, input_length, input_dim = input_shape
        #self.input = T.tensor3()

        if self.inner_rnn == 'gru':
            raise ValueError('this inner_rnn is not implemented yet. But it should be minor work, adjusting some initial state and stuff. try it yourself!')
            self.rnn = GRU(
                activation='relu',
                input_dim=input_dim+self.m_length,
                input_length=input_leng,
                output_dim=self.output_dim, init=self.init,
                inner_init=self.inner_init)
        elif self.inner_rnn == 'lstm':
            self.rnn = LSTM(
                output_dim = self.output_dim,
                implementation = 2,     # implemenation 0 seems to be a bit buggy regarding step function behavior
#                units = self.output_dim,
                input_shape = (bs, input_length, input_dim + self.m_length),
                #kernel_initializer=self.init,
                #inner_init=self.inner_init)
                #unit_forget_bias_init='zero'
                )
        else:
            raise ValueError('this inner_rnn is not implemented yet.')

#        self.rnn.build(input_shape)
        self.rnn.build(input_shape=(bs, input_length, input_dim + self.m_length))

        # WARNING: Not understood, only copied from keras/recurrent.py
        # In our case the dimension seems to be 5 (LSTM) or 4 (GRU),
        # see get_initial_states, those respond to:
        # init_M, init_wr, init_ww, init_h, init_c (LSTM only)
        self.states = [None, None, None, None, None]

        # initial memory, state, read and write vectors
        #
        self.M = K.variable(value=(.001*np.ones((1,))), name="main_memory")
        self.init_h = K.zeros(shape=(1,self.output_dim), name="state_vector")
        self.init_wr = K.variable(np.ones(self.n_slots)/self.n_slots, name="read_vector")
        self.init_ww = K.variable(np.ones(self.n_slots)/self.n_slots, name="write_vector")

        # write: erase, then add.
        self.W_e = self.rnn.kernel_initializer((self.output_dim, self.m_length))  # erase
        self.b_e = K.zeros((1,self.m_length), name="write_erase_bias")
        self.W_a = self.rnn.kernel_initializer((self.output_dim, self.m_length))  # add
        self.b_a = K.zeros((1,self.m_length), name="write_add_bias")

        #
        # get_w  parameters for reading operation
        #
        # key vector
        self.W_k_read = self.rnn.kernel_initializer((self.output_dim, self.m_length))
        self.b_k_read = self.rnn.bias_initializer((1, self.m_length))
        # 3 continuos(!) parameters, beta, g, gamme, as referenced in Figure 2 respectivly
        # equations 5, 7, 9
        self.W_c_read = self.rnn.kernel_initializer((self.output_dim, 3))
        self.b_c_read = self.rnn.bias_initializer((1,3))
        # shift 
        self.W_s_read = self.rnn.kernel_initializer((self.output_dim, self.shift_range))
        self.b_s_read = self.rnn.bias_initializer((1,self.shift_range)) 

        #
        # get_w  parameters for writing operation
        #
        # key vector
        self.W_k_write = self.rnn.kernel_initializer((self.output_dim, self.m_length))
        self.b_k_write = self.rnn.bias_initializer((1, self.m_length))
        # 3 continuos(!) parameters, beta, g, gamme, as referenced in Figure 2 respectivly
        # equations 5, 7, 9
        self.W_c_write = self.rnn.kernel_initializer((self.output_dim, 3))
        self.b_c_write = self.rnn.bias_initializer((1,3))
        # shift 
        self.W_s_write = self.rnn.kernel_initializer((self.output_dim, self.shift_range))
        self.b_s_write = self.rnn.bias_initializer((1,self.shift_range))

        self.C = _circulant(self.n_slots, self.shift_range)

        self.trainable_weights = self.rnn.trainable_weights + [
            self.W_e, self.b_e,
            self.W_a, self.b_a,
            self.W_k_read, self.b_k_read,
            self.W_c_read, self.b_c_read,
            self.W_s_read, self.b_s_read,
            self.W_k_write, self.b_k_write,
            self.W_s_write, self.b_s_write,
            self.W_c_write, self.b_c_write,
            self.M,
            self.init_h, self.init_wr, self.init_ww]

        if self.inner_rnn == 'lstm':
            self.init_c = K.zeros((1,self.output_dim), name="init_controller")
            self.trainable_weights = self.trainable_weights + [self.init_c, ]

        super(NeuralTuringMachine, self).build(input_shape)

    def _read(self, w, M):
        return K.sum((w[:, :, None]*M),axis=1)

    def _write(self, w, e, a, M):
        Mtilda = M * (1 - w[:, :, None]*e[:, None, :])
        Mout = Mtilda + w[:, :, None]*a[:, None, :]
        return Mout

    # See chapter 3.3.1
    def _get_content_w(self, beta, k, M):
        num = beta[:, None] * _cosine_distance(M, k)
        return K.softmax(num) #it was _softmax before, but that does the same?

    # This is as described in chapter 3.2.2
    def _get_location_w(self, g, s, C, gamma, wc, w_tm1):
        # Equation 7:
        wg = (g[:, None] * wc) + (1-g[:, None])*w_tm1
        # Cs is the circular convolution
        Cs = K.sum((C[None, :, :, :] * wg[:, None, None, :]),axis=3)
        # Equation 8:
        wtilda = K.sum((Cs * s[:, :, None]),axis=1)
        # Equation 9:
        wout = _renorm(wtilda ** gamma[:, None])
        return wout

    def _get_controller_output(self, h, W_k, b_k, W_c, b_c, W_s, b_s):
        k = K.tanh(K.dot(h, W_k) + b_k)  # + 1e-6
        c = K.dot(h, W_c) + b_c
        beta = K.relu(c[:, 0]) + 1e-4
        g = K.sigmoid(c[:, 1])
        gamma = K.relu(c[:, 2]) + 1.0001
        s = K.softmax(K.dot(h, W_s) + b_s)
        return k, beta, g, gamma, s

    def get_initial_state(self, X):
        batch_size = K.int_shape(X)[0]
        #FIXME! 
        batch_size = self.batch_size
#       # only commented the following out because its so hillariously bonkers.
#        init_M = self.M.dimshuffle(0, 'x', 'x').repeat(
#            batch_size, axis=0).repeat(self.n_slots, axis=1).repeat(
#            self.m_length, axis=2)        
#        init_M = init_M.flatten(ndim=2)  
        
        #FIXME: Why is the memory flattened at all, only to be unflattened later?
        init_M = K.ones((batch_size, self.n_slots * self.m_length))*0.001
#       # the bonkers code from above, in case its actually necessary (metadata)        
#        init_M = K.batch_flatten(K.repeat(K.repeat(K.repeat(K.FIXME ,
#                                                            batch_size, axis=0),
#                                                            self.n_slots, axis=1),
#                                                            self.m_length, axis=2))
        init_h = K.variable(np.zeros((batch_size, self.output_dim)), name="init_h")
        init_wr = K.ones((batch_size, self.n_slots), name="init_wr")/self.n_slots
        init_ww = K.ones((batch_size, self.n_slots), name="init_wr")/self.n_slots
        if self.inner_rnn == 'lstm':
            init_c = K.repeat_elements(self.init_c, batch_size, axis=0)
            return [init_M, K.softmax(init_wr), K.softmax(init_ww), init_h, init_c] #the softmax here confuses me.
        else:
            return [init_M, K.softmax(init_wr), K.softmax(init_ww), init_h]

    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.return_sequences:
            return input_shape[0], input_shape[1], self.output_dim
        else:
            return input_shape[0], self.output_dim

    def get_full_output(self, train=False):
        """
        This method is for research and visualization purposes. Use it as
        X = model.get_input()  # full model
        Y = ntm.get_output()    # this layer
        F = theano.function([X], Y, allow_input_downcast=True)
        [memory, read_address, write_address, rnn_state] = F(x)

        if inner_rnn == "lstm" use it as
        [memory, read_address, write_address, rnn_cell, rnn_state] = F(x)

        """
        # input shape: (nb_samples, time (padded with zeros), input_dim)
        X = self.get_input(train)
        assert K.ndim(X) == 3
        if K._BACKEND == 'tensorflow':
            if not self.input_shape[1]:
                raise Exception('When using TensorFlow, you should define ' +
                                'explicitely the number of timesteps of ' +
                                'your sequences. Make sure the first layer ' +
                                'has a "batch_input_shape" argument ' +
                                'including the samples axis.')

        mask = self.get_output_mask(train)
        if mask:
            # apply mask
            X *= K.cast(K.expand_dims(mask), X.dtype)
            masking = True
        else:
            masking = False

        if self.stateful:
            initial_states = self.states
        else:
            initial_states = self.get_initial_states(X)

        states = rnn_states(self.step, X, initial_states,
                            go_backwards=self.go_backwards,
                            masking=masking)
        return states

#    def call(self, input):
        # split input

        # read from memory by the instructions given in the last step

        # let the controller do its work

        # write to memory

        # give the new read instructions

        # return 
#        return None

    def step(self, inputs, states):
        #import pudb; pu.db


        M_tm1, wr_tm1, ww_tm1 = states[:3]
        # reshape
        #FIXME! how do we get the batchsize here?
        # self.batch_size is a temporary solution
        M_tm1 = K.reshape(M_tm1,(self.batch_size, self.n_slots, self.m_length))
        # read
        h_tm1 = states[3:]
        k_read, beta_read, g_read, gamma_read, s_read = self._get_controller_output(
            h_tm1[0], self.W_k_read, self.b_k_read, self.W_c_read, self.b_c_read,
            self.W_s_read, self.b_s_read)
        wc_read = self._get_content_w(beta_read, k_read, M_tm1)
        wr_t = self._get_location_w(g_read, s_read, self.C, gamma_read,
                                    wc_read, wr_tm1)
        M_read = self._read(wr_t, M_tm1)

        # update controller
        h_t = _update_controller(self, inputs, h_tm1, M_read)

        # write
        k_write, beta_write, g_write, gamma_write, s_write = self._get_controller_output(
            h_t[0], self.W_k_write, self.b_k_write, self.W_c_write,
            self.b_c_write, self.W_s_write, self.b_s_write)
        wc_write = self._get_content_w(beta_write, k_write, M_tm1)
        ww_t = self._get_location_w(g_write, s_write, self.C, gamma_write,
                                    wc_write, ww_tm1)
        # erase
        e = K.sigmoid(K.dot(h_t[0], self.W_e) + self.b_e)
        # add
        a = K.tanh(K.dot(h_t[0], self.W_a) + self.b_a)
        M_t = self._write(ww_t, e, a, M_tm1)

        M_t = K.batch_flatten(M_t)

        return h_t[0], [M_t, wr_t, ww_t] + h_t
