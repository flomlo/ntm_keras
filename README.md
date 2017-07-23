This code tries to implement the Neural Turing Machine, as found in 
https://arxiv.org/abs/1410.5401, as a backend neutral recurrent keras layer.



NOTE:
* For debugging purposes, the backend is currently restricted to tensorflow, but it should work for every backend by
commenting out those line which rely on it (its really only debug, and tensorboard).
* You may want to change the LOGDIR_BASE in testing_utils.py to something that works for you.
* The implementation with LSTM-controller is currently highly sensitive to the implementation of the LSTM-layer in keras
  itself. Therefore, it may break occasionally with keras updates.





For a quick start, type 

    python main.py ntm

while in a python enviroment which has tensorflow, keras, numpy. tensorflow-gpu is recommend, as everything is about 10x
faster.

This builds a NTM with *one* dense layer of appropriate size, tries it on the copyTask for length 5,10,20,40,80 and then
trains it on 1 million samples (1000 epochs of each 1000 freshly generated samples length between 5 and 20, adjust that
in main.py), which takes about 3h a GTX 1050Ti. 

For me that resulted in 50% accuracy (seen bitwise) before training, and 83%, 93%, 93%, 81%, 54% after training. Which
is success, but not huge success.



Why is it success? Compare it with just the controller alone, i.e a singe units=8 dense layer in temporal wrapping:
This can be tested via:

    python main.py dense 

And results in what is to be expected: 50% before, 50% after training. At least
the training was quick, taking only one minute (with the PCI-Express being the
bottleneck). 




But why not huge success? Well, besides the obvious fact that even 80% on length 5 is not at all great,
You may compare that with 3 layers of LSTMs:

    python main.py lstm

This builds 3 layers of LSTM with size 256 each (see Table 3 of the paper) and goes through the same testing procedure
as above, which for me resulted in 50% (before trainig), a training time of approximately 1h (same GPU) and 
(roughly) 100%, 100%, 94%, 50%, 50% accuracy at the respective test lengths. 

