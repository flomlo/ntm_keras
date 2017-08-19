import argparse


from keras.layers.core import Dense
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.optimizers import Adam, SGD
from keras.initializers import RandomNormal

output_dim = 8
input_dim = output_dim + 2  # this is the actual input dim of the network, that includes two dims for flags
batch_size = 100

controller_input_dim = 30
controller_output_dim = 100
#testrange=[5,10,20,40,80,160]


parser = argparse.ArgumentParser()
parser.add_argument("modelType", help="The kind of model you want to test, either ntm, dense or lstm")
parser.add_argument("-e", "--epochs", help="The number of epochs to train", default="1000", type=int)
parser.add_argument("-c", "--ntm_controller_architecture", help="""Valid choices are: dense, double_dense or
                                    lstm. Ignored if model is not ntm""", default="dense")
parser.add_argument("-v", "--verboose", help="""Verboose training: If enabled, the model is evaluated extensively
                                    after each training epoch.""", action="store_true")
args = parser.parse_args()
modelType = args.modelType
epochs = args.epochs
ntm_controller_architecture = args.ntm_controller_architecture
verboose = args.verboose

lr = 5e-4
clipnorm = 10
sgd = Adam(lr=lr, clipnorm=clipnorm)
sameInit = RandomNormal(seed=0)

if modelType == 'lstm':
    import model_lstm
    model = model_lstm.gen_model(input_dim=input_dim, output_dim=output_dim, batch_size=batch_size)

elif modelType == 'dense':
    import model_dense
    model = model_dense.gen_model(input_dim=input_dim, output_dim=output_dim, batch_size=batch_size)

elif modelType == 'ntm':
    import model_ntm

    controller = Sequential()
    controller.name=ntm_controller_architecture
    if ntm_controller_architecture == "dense":
        controller.add(Dense(units=controller_output_dim,
                                kernel_initializer=sameInit,
                                bias_initializer=sameInit,
                                activation='sigmoid',
                                input_dim=controller_input_dim))
    elif ntm_controller_architecture == "double_dense":
        controller.add(Dense(units=150,
                                kernel_initializer=sameInit,
                                bias_initializer=sameInit,
                                activation='sigmoid',
                                input_dim=controller_input_dim))
        controller.add(Dense(units=controller_output_dim,
                                kernel_initializer=sameInit,
                                bias_initializer=sameInit,
                                activation='sigmoid'))
    elif ntm_controller_architecture == "lstm":
        controller.add(LSTM(units=controller_output_dim,
                                kernel_initializer='random_normal', 
                                bias_initializer='random_normal',
                                activation='sigmoid',
                                stateful=True,
                                implementation=2,   # best for gpu. other ones also might not work.
                                input_shape=(batch_size, None, controller_input_dim)))
    else:
        raise ValueError("This controller_architecture is not implemented.")

    controller.compile(loss='binary_crossentropy', optimizer=sgd, metrics = ['binary_accuracy'], sample_weight_mode="temporal")
        
    model = model_ntm.gen_model(input_dim=input_dim, output_dim=output_dim, batch_size=batch_size,
                                    controller_model=controller)
else:
    raise ValueError("this model is not implemented")

print("model built, starting the copy experiment")
from testing_utils import lengthy_test
lengthy_test(model, epochs=epochs, verboose=verboose)
