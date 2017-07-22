import model_lstm

from testing_utils import lengthy_test

output_dim = 8
input_dim = output_dim+2
batch_size=100
testrange=[5,10,20,40,80]
epochs=1000


model_LSTM = model_lstm.gen_model(input_dim=input_dim, output_dim=output_dim, batch_size=batch_size)

lengthy_test(model_LSTM, epochs=epochs)


import pudb; pu.db
