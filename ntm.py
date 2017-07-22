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
# model by itself, not embeddable at all), which was his, might turn out to be visionary. Lets see.

# Oh, before I forget: In the end, I consider copyright to this code with exception to _circulant, considering the other
# helper functions as prior art (which was in the NTM paper), to lay with me. As far as copyright for some
# implementation for some idea found in some paper can lay in anybodies hands.
#
# If I understand correctly, the perfect License for "leave me alone I dont care" would be BSD 3 (which is the same for
# EderSantanas original code, just in doubt).
#
# Therefore, this code is licensed under BSD v3.
#



import numpy as np
import tensorflow as tf

import keras
from keras.layers.recurrent import Recurrent, GRU, LSTM
from keras.layers.core import Dense
from keras.initializers import RandomNormal, Orthogonal, Zeros, Constant
from keras import backend as K
from keras.engine.topology import InputSpec 

def _circulant(leng, n_shifts):
    # This is more or less the only code still left from the original author,
    # EderSantan @ Github.
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
    # this is equation (6), or as I like to call it: The NaN factory.
    # TODO: Find it in a library (cosine loss?)
    nk = K.l2_normalize(k, axis=-1)
    nM = K.l2_normalize(M, axis=-1)
    cosine_distance = K.batch_dot(nM, nk)
    #cosine_distance = tf.Print(cosine_distance, [nM[0], nk[0], cosine_distance[0]], message="_cosine_distance_numbers")
    return cosine_distance


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
                        controller_architecture='dense',
                        batch_size=777,                 
#                        input_shape = (None, 8), 
                        **kwargs):
        self.output_dim = output_dim
        self.units = output_dim
        self.n_slots = n_slots
        self.m_length = m_length
        self.shift_range = shift_range
        self.controller_architecture = controller_architecture
        self.batch_size = batch_size
        self.return_sequence = True
        # For calculating the controller output dimension, we need the output_dim of the whole layer
        # (which is only passed during building) plus all the stuff we need to interact with the memory,
        # calculated here:
        #
        # For every read head the addressing data (for details, see figure 2):
        #       key_vector (m_length) 
        #       beta (1)
        #       g (1)
        #       shift_vector (shift_range)
        #       gamma (1)
        self.controller_read_head_emitting_dim = (
                m_length + 1 + 1 + shift_range + 1)
        # But what do for write heads? The adressing_data_dim is the same, but we emit additionally:
        #       erase_vector (m_length)
        #       add_vector (m_length)
        self.controller_write_head_emitting_dim = (
                        self.controller_read_head_emitting_dim + 
                        2 * m_length)

        super(NeuralTuringMachine, self).__init__(**kwargs)

        


    def build(self, input_shape):
        bs, input_length, input_dim = input_shape

        # The controller output size:
        # TODO: arbitrary number of write/read heads
        self.controller_output_dim = (self.output_dim + 
                self.controller_read_head_emitting_dim + 
                self.controller_write_head_emitting_dim)
        # For the input shape of the controller the formula is a bit easier:
        #       the regular input_dim (output_dim)
        # plus, for every read head:
        #       read_vector (m_length).
        # So that results in:
        # TODO: arbitrary number of read heads
        self.controller_input_dim = input_dim + self.m_length

        # Now that we've calculated the shape of the controller, we have add it to the layer/model.
        if self.controller_architecture == 'gru':
            raise ValueError('This controller_architecture is not implemented yet. But it should be minor work, adjusting some initial state and stuff. try it yourself!')
            self.controller = GRU(
                name = "controller"
                #TODO
                )
        elif self.controller_architecture == 'lstm':
            self.controller = LSTM(
                name = "controller",
                units= self.controller_output_dim,
                implementation = 2,     # implemenation 0 seems to be a bit buggy regarding step function behavior
                input_shape = (bs, input_length, self.controller_input_dim))
        elif self.controller_architecture is 'dense':
            self.controller = Dense(
                name = "controller",
                activation = 'sigmoid',
                bias_initializer = 'zeros',
                units = self.controller_output_dim,
                input_shape = (bs, input_length, self.controller_input_dim))
        else:
            # FEATURE REQUEST: Handling a whole Keras *model* as controller would be so incredible cool.
            raise ValueError('this controller_architecture is not implemented yet.')

        self.controller.build(input_shape=(bs, input_length, input_dim + self.m_length))

        self.C = _circulant(self.n_slots, self.shift_range)

        self.trainable_weights += self.controller.trainable_weights 
        # By declaring our memory as a weight (non-trainable), we can easily look at it via model.get_weigths.
        self.M = self.add_weight(shape=(self.batch_size, self.n_slots, self.m_length),
                                    name="Main_Memory",
                     #               initializer=Constant(self.m_length**-0.5), # better cosine stabily
                                    initializer=Constant(0.001),
                                    trainable=False)

        # We need to declare the number of states we want to carry around.
        # In our case the dimension seems to be 6 (LSTM) or 5 (GRU) or 4 (FF),
        # see self.get_initial_states, those respond to:
        # [old_ntm_output] + [init_M, init_wr, init_ww] +  [init_h] (LSMT and GRU) + [(init_c] (LSTM only))
        # old_ntm_output does not make sense in our world, but is required by the definition of the step function we
        # intend to use.
        # WARNING: What self.state_spec does is only poorly understood,
        # I only copied from keras/recurrent.py.
        self.states = [None, None, None, None]
        self.state_spec = [InputSpec(shape=(None, self.output_dim)),                            # old_ntm_output
                            InputSpec(shape=(None, self.n_slots * self.m_length)),              # Memory
                            InputSpec(shape=(None, self.n_slots)),                              # init_wr
                            InputSpec(shape=(None, self.n_slots))]                              # init_ww

        if self.controller_architecture in ['GRU', 'lstm']: 
            self.states += [None]
            self.state_spec += [InputSpec(shape=(None, self.controller_output_dim))]            # init_h (GRUS/LSTM)
        if self.controller_architecture is 'lstm':
            self.states += [None]
            self.state_spec += [InputSpec(shape=(None, self.controller_output_dim))]            # init_c (LSTM only)

        super(NeuralTuringMachine, self).build(input_shape)


    def get_initial_state(self, X):
        batch_size = K.int_shape(X)[0]
        #FIXME! make batchsize variable, not fixed with model 
        batch_size = self.batch_size
        

        init_old_ntm_output = K.ones((batch_size, self.output_dim), name="init_old_ntm_output")*0.42 # never used.
        #init_M = K.ones((batch_size, self.n_slots , self.m_length), name='main_memory')*self.m_length**-0.5
        init_M = self.M
        init_wr = np.zeros((batch_size, self.n_slots))
        init_wr[:,0] = 1    # turns out that uniform initialisation is almost perfectly unperfect.
        init_wr = K.variable(init_wr, name="init_weights_read")
        init_ww = np.zeros((batch_size, self.n_slots))
        init_ww[:,0] = 1
        init_ww = K.variable(init_ww, name="init_weights_write")
        
        if self.controller_architecture is 'dense':   # actually: stateless
            return [init_old_ntm_output, init_M, init_wr, init_ww]
        else:
            init_h = K.ones((batch_size, self.controller_output_dim), name="init_h")*0.5
            if self.controller_architecture is 'GRU':
                raise ValueError('not implemented yet')
                return [init_old_ntm_output, init_M, init_wr, init_ww, init_h]
            elif self.controller_architecture is "lstm":
                init_c = K.ones((batch_size, self.controller_output_dim), name="init_c")*0.5
                return [init_old_ntm_output, init_M, init_wr, init_ww, init_h, init_c]
            else:
                raise ValueError('not implemented yet')




    # See chapter 3.1
    def _read(self, weights, M):
        return K.sum((weights[:, :, None]*M),axis=1)

    # See chapter 3.2
    def _write_to_memory(self, M, w, e, a):
        # see equation (3)
        M_tilda = M * (1 - w[:, :, None]*e[:, None, :])
        # see equation (4)
        Mout = M_tilda + w[:, :, None]*a[:, None, :]
        #Mout = tf.Print(Mout, [Mout[0], M[0], w[0], e[0], a[0]], message="_write_to_memory")
        return Mout

    # This is the chain described in Figure 2, or in further detail by
    # Chapter 3.3.1 (content based) and Chapter 3.3.2 (location based)
    # C is our convolution function precomputed above.
    def _get_weight_vector(self, w_tm1, M, k, beta, g, s, gamma):
        # Content adressing, see Chapter 3.3.1:
        num = beta * _cosine_distance(M, k)
        w_c  = K.softmax(num) # It turns out that equation (5) is just softmax.
        # Location adressing, see Chapter 3.3.2:
        # Equation 7:
        w_g = (g * w_c) + (1-g)*w_tm1
        # C_s is the circular convolution
        #C_w = K.sum((self.C[None, :, :, :] * w_g[:, None, None, :]),axis=3)
        # Equation 8:
        # TODO: Explain
        C_s = K.sum(K.repeat_elements(self.C[None, :, :, :], self.batch_size, axis=0) * s[:,:,None,None], axis=1)
        w_tilda = K.batch_dot(C_s, w_g)
        # Equation 9:
        w_out = _renorm(w_tilda ** gamma)

        w_out = tf.Print(w_out, [w_tm1[0], M[0], k[0], beta[0], s[0], gamma[0]], message="_get_weight_vectors inputs: w_tm1, M, k, beta, s, gamma ")
        w_out = tf.Print(w_out, [num[0], w_c[0], w_g[0], C_s[0], w_tilda[0], w_out[0]], message="_get_weight_vectors calculations:num, w_c,  w_g, C_s, w_tilda, w_out ")
        w_out = tf.Print(w_out, [K.sum(w_c[0], axis=-1), K.sum(w_g[0], axis=-1), K.sum(w_tilda[0], axis=-1), K.sum(w_out[0], axis=-1)], message="_get_weight_vectors sum calculations: If all one, all is good. ")
        
        lowerBoundAlert = tf.assert_non_negative(w_out, message="weights vector had value < 0")
        upperBoundAlert = tf.assert_non_positive(w_out - 1, message="weights vector had value > 1")
        with tf.control_dependencies([lowerBoundAlert, upperBoundAlert]):
            w_out = -w_out
            w_out = -w_out

        return w_out

    def _run_controller(self, inputs, read_vector, controller_state):
        # FIXME: Could we spare ourself the hassle by setting the LSTM statefullness to True?
        #Warning: This is highly sensitive to the implementation of the LSTM, which could change. For example, it
        #        already breaks if we implementation 0 instead of implementation 2.
        controller = self.controller
        controller_input = K.concatenate([inputs, read_vector])
        # begin magic: (inspired by EderSantana)
        # TODO: I'm quite sure this could be done less implementation-sensitive.
        if len(controller_state) in [1,2]: #LSTM or GRU
            if hasattr(controller, "get_constants"):
                BW,BU = controller.get_constants(controller_input)
                controller_state += (BW,BU)
                controller_output, controller_state = controller.step(controller_input, controller_state)
        else: # dense
            controller_output = controller.call(controller_input)
        return controller_output, controller_state




    @property
    def output_shape(self):
        input_shape = self.input_shape
        if self.return_sequences:
            return input_shape[0], input_shape[1], self.output_dim
        else:
            return input_shape[0], self.output_dim


    def step(self, inputs, states):
        # As a step function MUST return its regular output as the first element in the list of states,
        # we have _ here.
        _, M_tm1, w_read_tm1, w_write_tm1 = states[:4]
        controller_states = list(states[4:]) #this could be empty (FF), length 1 (GRU) or length 2 (LSTM)
        


        # We have the Memory M_tm1 (t minus one / t-1), and a read weighting w_read_tm1 calculated in the last
        # step. This is enough to calculate the read_vector we feed into the controller:
        read_vector = self._read(w_read_tm1, M_tm1)

        # Now feed the controller and let it run a single step, implemented by calling the step function directly,
        # which we have to provide with the actual input from outside, the information we've read an the states which
        # are relevant to the controller.
        # TODO make description more precise
        controller_output, controller_states = self._run_controller(inputs, read_vector, controller_states)

        # Now we split the controller output into actual output, read head adressing, write head adressing, etc.
        # TODO: make more precise
        # We now carefully split the gigantic output of the controller into rough chunks:
        # As a reminder, it is split up like that:
        # output_dim + controller_read_emitting_dim + controller_write_emitting_dim
        # TODO: Multiple heads
        # TODO: Im not very prout of that code. Not at all. Please improve it or help me improving it :(
        # TODO: At least put it into a separate function.

        ntm_output = controller_output[:, :self.output_dim]
        
        controller_read_emitted_data   = controller_output[:, self.output_dim : (self.output_dim +
                                                                                    self.controller_read_head_emitting_dim)]
        controller_write_emitted_data  = controller_output[:, -self.controller_write_head_emitting_dim :]
        
        # Now for fine grained output
        # The further naming follows the naming of the NTM-Paper, but we note if its for reading or writing: 

        # We also shift our values so that they are near the most "neutral" value, asuming they were 0 before (as is the
        # bias for the controller network),
        # e.g. so that k is normalized (good for numerical stabilty for cosine similarity) or
        # or g is 0.5, which is a more reasonable starting value for a convex combinator, 
        # or gamma >= 1 (which it has to be anyway according to the paper)

        # We asume that our default value is 0.5 for everything, as it is sigmoid(0).

        k_read          = controller_read_emitted_data[:, : self.m_length] 
        k_read          += self.m_length**-0.5 - 0.5
        beta_read       = controller_read_emitted_data[:, self.m_length : self.m_length + 1] 
        beta_read       += 0.5 #actually I have no clue what a good magic value would be for that, but 1 is multiplication neutral
        g_read          = controller_read_emitted_data[:, self.m_length + 1: self.m_length + 1 + 1] 
        g_read          += 0.0  
        shift_read      = controller_read_emitted_data[:, self.m_length + 1 + 1 : self.m_length + 1 + 1 + self.shift_range]
        shift_read      = K.softmax(shift_read)  # normalize it via softmax
        gamma_read      = controller_read_emitted_data[:, -1 :]
        gamma_read      = K.clip(5* gamma_read +0.5, 1, 10) 

        k_write         = controller_write_emitted_data[:, : self.m_length] 
        k_write         += self.m_length**-0.5 -0.5
        beta_write      = controller_write_emitted_data[:, self.m_length : self.m_length + 1] 
        beta_write      += 0.5
        g_write         = controller_write_emitted_data[:, self.m_length + 1: self.m_length + 1 + 1] 
        g_write         += 0.0  
        shift_write     = controller_write_emitted_data[:, self.m_length + 1 + 1 : self.m_length + 1 + 1 + self.shift_range]
        shift_write     = K.softmax(shift_write)  # normalize it via softmax
        gamma_write     = controller_write_emitted_data[:, self.m_length + 1 + 1 + self.shift_range : - 2*self.m_length]
        gamma_write     = K.clip(5*gamma_write +0.5, 1, 5) 
        erase_vector    = controller_write_emitted_data[:, -2*self.m_length : -self.m_length]
        erase_vector    += -0.5 # zero is fine
        add_vector      = controller_write_emitted_data[:, -self.m_length : ]
        add_vector      += -0.5 # zero is fine


        # Now we want to write to the memory.
        # But first, we have to calculate the adress we send the erase and add vektor to:
        # As seen in Figure 2 of the Paper, this depends on a lot of variables:
        w_write = self._get_weight_vector (w_write_tm1, M_tm1, k_write, beta_write, g_write, shift_write, gamma_write)
        # w_write = tf.Print(w_write_tm1, [w_write[0], w_write_tm1[0]], message="write weights: new, old:")

        # But now we can manipulate it, using old Memory, the w_write adress vector, erase and add vector:
        M = self._write_to_memory(M_tm1, w_write, erase_vector, add_vector)
        #ntm_output = tf.Print(ntm_output, [M[0], M_tm1[0]], message="The Memory looks like: ")

        # Only one thing left until this step is complete: Calculate the read weights we save in the state and use next
        # round:
        w_read = self._get_weight_vector (w_read_tm1, M, k_read, beta_read, g_read, shift_read, gamma_read)

        # Now lets pack up the state in a list and call it a day.
        # ntm_output = tf.Print(ntm_output, [gamma_read, gamma_write], message="gamma_read, gamma_write")
        return ntm_output, [ntm_output] + [M, w_read, w_write] + controller_states









# PS: As always: Dont forget to bind your AIs energy supply to the condition of you being still alive.
