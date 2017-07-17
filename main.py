from keras.utils import plot_model
import ntmTest
model_NTM, batch_size = ntmTest.gen_model()

plot_model(model_NTM, to_file='graphs/model_NTM.png', show_shapes = True)

from copyTask import lengthy_test

lengthy_test(model_NTM, batch_size=batch_size, training_epochs=1000)
import pudb; pu.db
