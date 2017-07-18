# History of this code:
# This is all based on ntm.py found in the seya-repository of EderSantana, 
# https://github.com/EderSantana/seya/blob/master/seya/layers/ntm.py
# as (git clone)'d on 2017-07-12 or something like that.
#
# Problem: the code was written for keras 0.12 (?), and relied heavily on Theano as the backend.
#
# This code was then reworked by me, and by reworking I mean: cluelessly debugging, trying to understand what it does,
# even looking up ancient keras documentation in the wayback machine (*cries*), rewriting all that to something which seems
# like the semantic equivalence in Keras 2, Backend neutral code.
# That would have gone quicker if I've either understood Keras, Theno, Python debugging or what the NTM even does.
#
# After that, and hours of debugging numerical features due to wrong initialization of vectors it then finally worked.
# By working I mean: All the actual interaction with the memory was completly broken. Just didnt work, behaved like a
# layer consisting solely like a LSTM (which makes quite a lot of sense)
#
# After a few days it dawned me that quite a lot of the topological descriptions were utter failures. Maybe I interpreted
# the original code wrong, or there were even bigger changes in the API than I ever understood. 
# 
# After that I rewrote almost everything that was left. The only thing semantically original are the helper functions
# which do *exactly* what is described in the NTM-Paper, and the _circulant function, which is actually quite nice!
# 
# But all in all, I really want to thank EderSantana. It was a shitload of work, but without his code to work on, and
# inproving in small steps until I really understood what I was doing, I would have never tried anyway.
#
# Also the overall idea of understanding the NTM as a recurrent layer to be used inside another model (and not solely as
# model by itself, not embeddable at all), which was his, might turn out to be visionary.
# Lets see.

# Oh, before I forget: In the end, I consider copyright to this code with exception to _circulant, considering the other
# helper functions as prior art (which was in the NTM paper), to lie with me.
#
# If I understand correctly, the perfect License for "leave me alone I dont care" would be BSD 3 (which is the same for
# EderSantanas original code, just in doubt).
#
# Therefore, this code is licensed by BSD v3.
#



import numpy as np

from keras.layers.recurrent import Recurrent, GRU, LSTM
from keras.initializers import Orthogonal, Zeros
from keras import backend as K
from keras.engine.topology import InputSpec 

from utils import rnn_states


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


    return h


def _circulant(leng, n_shifts):
    # This is more or less the only code still left from the original author,
    # EderSantan @ Github.
    # It works perfectly and is elegant! So grats to him.
    # My implementation would probably just be worse.
    # Below his original comment:
 
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
    Currently batch_input_size is necessary. Or not? Im not even sure :(

    """
    def __init__(self, output_dim, n_slots, m_length, shift_range=3,
                        inner_rnn='lstm',
                        batch_size=777,                 
                        **kwargs):
        super(NeuralTuringMachine, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.units = output_dim
        self.n_slots = n_slots
        self.m_length = m_length
        self.shift_range = shift_range
        self.inner_rnn = inner_rnn
        self.batch_size = batch_size
        # Calculate the controller output dimension, consisting of:
        #       the regular output dimension (output_dim)
        #
        # plus for every read head the addressing data (for details, see figure 2):
        #       key_vector (m_length) 
        #       beta (1)
        #       g (1)
        #       shift_vector (shift_range)
        #       gamma (1)
        self.controller_read_head_emitting_dim =
                m_length + 1 + 1 + shift_range + 1
        # But what do for write heads? The adressing_data_dim is the same, but we emit additionally:
        #
        #       erase_vector (m_length)
        #       add_vector (m_length)
        self.controller_write_head_emitting_dim =
                self.controller_read_head_emitting_dim +
                2 * m_length
        # So this results in:
        # TODO: arbitrary number of write/read heads
        self.controller_output_dim = self.output_dim + 
                self.controller_read_head_emitting_dim +
                self.controller_write_head_emitting_dim

        # For the input shape of the controller the formula is a bit easier:
        #       the regular input_dim (output_dim)
        # plus, for every read head:
        #       read_vector (m_length).
        # So that results in:
        # TODO: arbitrary number of read heads
        self.controller_input_dim = 
                self.input_dim +
                m_length

        
        # WARNING: Only poorly understood, only copied from keras/recurrent.py
        # In our case the dimension seems to be 5 (LSTM) or 4 (GRU / FeedForward),
        # see self.get_initial_states()
        self.state_spec = [InputSpec(shape=(None, self.n_slots * m_length)),         # Memory
                            InputSpec(shape=(None, self.n_slots)),                   # init_wr
                            InputSpec(shape=(None, self.n_slots)),                   # init_ww
                            InputSpec(shape=(None, self.output_dim)),                # init_h TODO: WTF is this actually?
                            InputSpec(shape=(None, self.output_dim))]                # init_c (LSTM only)


    def build(self, input_shape):
        bs, input_length, input_dim = input_shape

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
                units= self.controller_output_dim,
                implementation = 2,     # implemenation 0 seems to be a bit buggy regarding step function behavior
                input_shape = (bs, input_length, self.controller_input_dim))
        else:
            # TODO: FeedForward Network (Just a dense layer) would be a must-have feature.
            # FEATURE REQUEST: However, handling a whole Keras *model* would be incredible cool.
            raise ValueError('this inner_rnn is not implemented yet.')

#        self.rnn.build(input_shape)
        self.rnn.build(input_shape=(bs, input_length, input_dim + self.m_length))

        # WARNING: Only poorly understood, only copied from keras/recurrent.py
        # In our case the dimension seems to be 5 (LSTM) or 4 (GRU),
        # see get_initial_states, those respond to:
        # init_M, init_wr, init_ww, init_h, init_c (LSTM only)
        self.states = [None, None, None, None, None]

        self.trainable_weights += self.rnn.trainable_weights 

        if self.inner_rnn == 'lstm':
            self.init_c = K.zeros((1,self.output_dim), name="init_controller")
            self.trainable_weights = self.trainable_weights + [self.init_c, ]

        super(NeuralTuringMachine, self).build(input_shape)


    def get_initial_state(self, X):
        batch_size = K.int_shape(X)[0]
        #FIXME! make batchsize variable, not fixed with model 
        batch_size = self.batch_size
#       #   only commented the previous version because its so hillariously bonkers:
#        init_M = self.M.dimshuffle(0, 'x', 'x').repeat(
#            batch_size, axis=0).repeat(self.n_slots, axis=1).repeat(
#            self.m_length, axis=2)        
#        init_M = init_M.flatten(ndim=2)  
        
        #FIXME: Why is the memory flattened at all, only to be unflattened later?
        init_M = K.ones((batch_size, self.n_slots * self.m_length))*0.001
        init_wr = K.ones((batch_size, self.n_slots), name="init_wr")/self.n_slots
        init_ww = K.ones((batch_size, self.n_slots), name="init_wr")/self.n_slots
        if self.inner_rnn == 'lstm':
            init_c = K.repeat_elements(self.init_c, batch_size, axis=0)
            return [init_M, K.softmax(init_wr), K.softmax(init_ww), init_c] #the softmax here confuses me.
        else:
            return [init_M, K.softmax(init_wr), K.softmax(init_ww)]



    # See chapter 3.1
    def _read(self, weights, M):
        return K.sum((weights[:, :, None]*M),axis=1)

    # See chapter 3.2
    def _write(self, weights, e, a, M):
        # see equation (3)
        M_tilda = M * (1 - w[:, :, None]*e[:, None, :])
        # see equation (4)
        Mout = M_tilda + w[:, :, None]*a[:, None, :]
        return Mout

    # This is the chain described in Figure 2, or in further detail by
    # Chapter 3.3.1 (content based) and Chapter 3.3.2 (location based)
    def _get_weight_vector(self, w_tm1, M, k, beta, g, s, C, gamma, wc):
        # Content adressing, see Chapter 3.3.1:
        num = beta[:, None] * _cosine_distance(M, k)
        w_c K.softmax(num) 
        # Location adressing, see Chapter 3.3.2:
        # Equation 7:
        w_g = (g[:, None] * wc) + (1-g[:, None])*w_tm1
        # Cs is the circular convolution
        C_s = K.sum((C[None, :, :, :] * wg[:, None, None, :]),axis=3)
        # Equation 8:
        w_tilda = K.sum((Cs * s[:, :, None]),axis=1)
        # Equation 9:
        wout = _renorm(wtilda ** gamma[:, None])
        return wout

    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.return_sequences:
            return input_shape[0], input_shape[1], self.output_dim
        else:
            return input_shape[0], self.output_dim


    def step(self, inputs, states):
        # TODO: decide what to do if there is no controller state
        if self.inner_rnn = 'lstm':
            M_tm1, w_read_tm1, w_write_tm1, controller_state = states
        else:
            ValueError("sorry, not implemented yet")

        # reshape
        #FIXME! how do we get the batchsize here?
        # self.batch_size is a temporary solution
        # addendum: might not be necessary if memory saved properly, now has right shape automatically 
        M_tm1 = K.reshape(M_tm1,(self.batch_size, self.n_slots, self.m_length))

        # read: 
        # We have the Memory M_tm1 (t minus one / t-1), and a read weighting w_read_tm1 calculated in the last
        # step. This is enough to calculate the read_vector we feed into the controller:
        read_vector = self._read(wr_tm1, M_tm1)

        # Now feed the controller and let it run a single step, implemented by calling the step function directly,
        # which we have to provide with the actual input from outside, the information we've read an the states which
        # are relevant to the controller.
        # TODO make description more precise
        controller_output, controller_state = _run_controller_step(self, inputs, read_vector, controller_state)

        # Now we split the controller output into actual output, read head adressing, write head adressing, etc.
        # TODO: make more precise
        # We now carefully split the gigantic output of the controller into rough chunks:
        # As a reminder, it is split up like that:
        # output_dim + controller_read_emitting_dim + controller_write_emitting_dim
        # TODO: Multiple heads
        # TODO: Im not very prout of that code. Not at all. Please improve it or help me improving it :(
        # TODO: At least put it into a separate function.

        ntm_output = controller_output[:self.output_dim]
        
        controller_read_emitted_data   = controller_output[self.output_dim : (self.output_dim +
                                                                                    self.controller_read_emitting_dim)]
        controller_write_emitted_data  = controller_output[-self.controller_write_emitting_dim :]
        
        # Now for fine grained output
        # The further naming follows the naming of the NTM-Paper, but we note if its for reading or writing: 

        k_read         = controller_read_emitted_data[: m_length] 
        beta_read      = controller_read_emitted_data[m_length : m_length + 1] 
        g_read         = controller_read_emitted_data[m_length + 1: m_length + 1 + 1] 
        shift_read     = controller_read_emitted_data[m_length + 1 + 1 : shift_range]
        gamma_read     = controller_read_emitted_data[-1 : ]

        k_write        = controller_write_emitted_data[: m_length] 
        beta_write     = controller_write_emitted_data[m_length : m_length + 1] 
        g_write        = controller_write_emitted_data[m_length + 1: m_length + 1 + 1] 
        shift_write    = controller_write_emitted_data[m_length + 1 + 1 : shift_range]
        gamma_write    = controller_write_emitted_data[m_length + 1 + 1 + shift_range : - 2*m_length]
        erase_vector   = controller_write_emitted_data[-2*m_length : -m_length]
        add_vector     = controller_write_emitted_data[-m_length : ]

        # In hindsight I now understand why DeepMind build Sonnet. It should be much easier and cleaner in there.

        
        # Now we want to write to the memory.
        # But first, we have to calculate the adress we send the erase and add vektor to:
        # As seen in Figure 2 of the Paper, this depends on a lot of variables:
        w_write = _get_weight_vector (self, w_write_tm1, M_tm1, k_write, beta_write, g_write, shift_write, gamma_write)

        # But now we can manipulate it, using old Memory, the w_write adress vector, erase and add vector:
        M = _write_to_memory(M_tm1, w_write, erase_vector, add_vector)

        # Only one thing left until this step is complete: Calculate the read weights we save in the state and use next
        # round:
        w_read = _get_weight_vector (self, w_read_tm1, M, k_read, beta_read, g_read, shift_read, gamma_read)

        # Now lets pack up the state in a list and call it a day.
        return ntm_output, [M, w_read, w_write, controller_state]





# PS: As always: Dont forget to bind your AIs energy supply to the condition of you being still alive.
