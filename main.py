import datetime
import ntmTest
import keras
import numpy as np
from keras.utils import plot_model
from copyTask import get_sample

output_dim = 8
input_dim = output_dim + 2  # this is the actual input dim of the network, that includes two dims for flags
batch_size = 100
#testrange=[5,10,20,40,80,160]
epochs=1000



model_NTM, batch_size = ntmTest.gen_model(input_dim=input_dim, output_dim=output_dim, batch_size=batch_size)

lengthy_test(model_NTM, epochs=epochs)

import pudb; pu.db

