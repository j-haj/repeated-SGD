import logging
import logging.config
import datetime
import time
import yaml
import os
import numpy as np

from core.util.loaders import MNISTLoader, Data
from core.function.function import SingleLayerNetwork
from core.function.loss_function import SquareLoss
from core.optimizers.optimizers import SGD, SRGD

TRAIN = Data.TRAIN

# Setup logging
with open("logging.yaml", "r") as fd:
    config = yaml.safe_load(fd.read())

# Add timestamp to file
log_filename = config["handlers"]["file"]["filename"]
base, extension = os.path.splitext(log_filename)
today_date = datetime.datetime.today()
log_filename = "{}{}{}".format(
        base,
        today_date.strftime("_%Y_%m_%d_%HH-%M-%S"),
        extension)
config["handlers"]["file"]["filename"] = log_filename
logging.config.dictConfig(config)
logger = logging.getLogger(__name__)

class Timer:
    """Simple timing class"""

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start


def run_mnist(learning_rate, x_vals, labels):
    # Model hyperparameters
    batch_size = 32
    num_epochs = 100
    early_stop = 0.0001

    # Create approx functions
    # Note: The network used in LeCun 98 was a 10 node with 10 biases
    # network with an input size of 28*28
    input_dim = 28*28
    nn_f_sgd = SingleLayerNetwork(dim=input_dim,
                                  num_neurons=10)
    nn_f_srgd = SingleLayerNetwork(dim=input_dim,
                                   num_neurons=10)
    
    # Create loss function
    sgd_loss = SquareLoss(nn_f_sgd)
    srgd_loss = SquareLoss(nn_f_srgd)
    
    # Create optimizers
    sgd = SGD(loss_func=sgd_loss,
              approx_func=nn_f_sgd,
              learning_rate=learning_rate)

    srgd = SRGD(loss_func=srgd_loss,
                approx_func=nn_f_srgd,
                learning_rate=learning_rate,
                repeat_num=5)

    # Run optimization
    (sgd_iter, sgd_epoch, sgd_errors) = sgd.solve(
            data=x_vals,
            labels=labels,
            num_epochs=num_epochs,
            batch_size=batch_size,
            early_stop=early_stop)

    (srgd_iter, srgd_epoch, srgd_errors) = srgd.solve(
            data=x_vals,
            labels=labels,
            num_epochs=num_epochs,
            batch_size=batch_size,
            early_stop=early_stop)
def main():
    """Main function"""

    # Load MNIST training data
    data_loader = MNISTLoader("/data/mnist/")
    (img,label) = data_loader.load_data(TRAIN)

    # Train model
    run_mnist(0.001, img, label)

    # Test model on test data
    print("training complete")

if __name__ == "__main__":
    main()
