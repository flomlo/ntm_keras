import argparse

#import model_ntm
#import model_dense

output_dim = 8
input_dim = output_dim + 2  # this is the actual input dim of the network, that includes two dims for flags
batch_size = 100
#testrange=[5,10,20,40,80,160]


parser = argparse.ArgumentParser()
parser.add_argument("modelType", help="the kind of model you want to test, either ntm, dense or lstm")
parser.add_argument("-e", "--epochs", help="the number of epochs to train", default="1000", type=int)
parser.add_argument("-c", "--ntm_controller_architecture", help="lstm or dense or gru, ignored if model is not ntm", default="dense")
args = parser.parse_args()
modelType = args.modelType
epochs = args.epochs
ntm_controller_architecture = args.ntm_controller_architecture


if modelType == 'lstm':
    import model_lstm
    model = model_lstm.gen_model(input_dim=input_dim, output_dim=output_dim, batch_size=batch_size)
elif modelType == 'ntm':
    import model_ntm
    model = model_ntm.gen_model(input_dim=input_dim, output_dim=output_dim, batch_size=batch_size,
                                    controller_architecture=ntm_controller_architecture)
elif modelType == 'dense':
    import model_dense
    model = model_dense.gen_model(input_dim=input_dim, output_dim=output_dim, batch_size=batch_size)
else:
    raise ValueError("this model is not implemented yet")

print("model built, starting the copy experiment")
from testing_utils import lengthy_test
lengthy_test(model, epochs=epochs)


