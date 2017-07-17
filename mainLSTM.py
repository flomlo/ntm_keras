from keras.utils import plot_model

import model_lstm

model_LSTM, batch_size = model_lstm.gen_model()

plot_model(model_LSTM, to_file='graphs/model_LSTM.png', show_shapes = True)

from copyTask import lengthy_test

lengthy_test(model_LSTM, batch_size=batch_size, training_epochs=100)
import pudb; pu.db
