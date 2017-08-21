#The Neural Turing Machine

This code tries to implement the Neural Turing Machine, as found in 
https://arxiv.org/abs/1410.5401, as a backend neutral recurrent keras layer.

NOTE:
* There is a nicely formatted paper describing the rough idea of the NTM, implementation difficulties and which discusses the
  copy experiment. It is available here in the repository as The_NTM_-_Introduction_And_Implementation.pdf. 
* You may want to change the LOGDIR_BASE in testing_utils.py to something that works for you or just set a symbolic
  link.
* So far one read and one write head is hardcoded into the system. This will probably be fixed in a future update.


### User guide
For a quick start on the copy Task, type 

    python main.py -v ntm

while in a python enviroment which has tensorflow, keras, numpy. tensorflow-gpu is recommend, as everything is about 20x
faster. In my case this takes about 100 minutes on a NVIDIA GTX 1050 Ti.
The -v is optional and offers much more detailed information about the achieved accuracy, and also after every training
epoch.

Compare that with

    python main.py lstm

This builds 3 layers of LSTM with and goes through the same testing procedure
as above, which for me resulted in a training time of approximately 1h (same GPU) and 
(roughly) 100%, 100%, 94%, 50%, 50% accuracy at the respective test lengths. 

Have fun with the results!


### API
From the outside, this implementation looks like a regular recurrent layer in keras.
It has however a number of non-obvious parameters:

n_width: is the width of the memory matrix 

m_depth: is the depth of the memory matrix (be careful about increasing that to much, it has quadratic influence on the
number of trainable weights)

controller_model: this parameter allows you to place a keras model of appropriate shape as the controller. the
appropriate shape can be calculated via controller_input_output_shape. If None is set, a single dense layer will be
used. 


## TODO:
* Arbitrary number of read and write heads
* Support of masking, and maybe dropout, one has to reason about it theoretically first.
* support for get and set config to better enable model saving
* A bit of code cleaning: especially the controller output splitting is ugly as hell.
* Although it should be backend neutral, some testing with other backends might be nice.
* Support for arbitrary activation functions would be nice, currently restricted to sigmoid.
