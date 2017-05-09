import logging
import logging.config
import datetime
import yaml
import os
import time
import numpy as np

from core.function.function import LinearFunction
from core.function.loss_function import SquareLoss
from core.optimizers.optimizers import SGD, SRGD

from sklearn.cross_validation import train_test_split

# Setup logging
with open("logging.yaml","r") as fd:
    config = yaml.safe_load(fd.read())

# Add timestamp to file
log_dir = "logs/"
log_filename = config["handlers"]["file"]["filename"]
print(log_filename)
base, extension = os.path.splitext(log_filename)
today_date = datetime.datetime.today()
log_filename = "{}{}{}".format(
        log_dir,
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

def load_data(path=None, delim=":"):
    """Loads the covtype data from ``path``

    Parameters:
        * path: the path to the data file

    Return: a tuple containing the labels and feature data sets
    """

    # This data set has 581,012 data points and the feature space
    # has a dimension of 54
    y = np.zeros((581012, 1))
    X = np.zeros((581012, 54))
    with open(path, "r") as freader:
        for index, line in enumerate(freader):
            line_elements = line.strip().split()
            y[index] = float(line_elements[0])
            x_vals = list(map(lambda x: (int(x.split(delim)[0]),
                                    float(x.split(delim)[1])),
                          line_elements[1:]))
            x_row = np.zeros(54)
            for x in x_vals:
                x_row[x[0] - 1] = x[1]
            X[index] = x_row

    return (y, X)

def grad(diff, x_vals):
    return diff * x_vals

def fit_least_squares(labels, features):
    """Fits the data to a linear function using a square loss function."""
    results = {"sgd": [], "srgd": []}
   
    # Create test and train data
    n_train = 450000
    
    train_features, test_features, train_labels, test_labels = train_test_split(
            features, labels, test_size=0.2, random_state=42)

    #logger.info("Splitting data into training and test sets")
    #train_indices = np.random.choice(581012, n_train, replace=False)
    #logger.info("Training indices created")
    #train_labels = [labels[i] for i in train_indices]
    #logger.info("Training labels partitioned")
    #test_lables = [labels[i] for i in range(len(labels)) if i not in
    #        train_indices]
    #logger.info("Test labels partitioned")
    #train_features = [features[i, :] for i in train_indices]
    #logger.info("Training features partitioned")
    #test_features = [features[i, :] for i in range(len(labels)) if i not in
    #        train_indices]
    #logger.info("Test features partitioned")
    #logger.info("Training and test sets created...")

    # Hyperparameters
    dim = 54
    num_tests = 1
    mini_batch_size = 10
    r = 3
    learning_rate = 0.01
    scaler = 1
    n_epochs = 10

    approx_f_sgd = LinearFunction(dim=dim)
    approx_f_srgd = LinearFunction(dim=dim)

    sgd_loss = SquareLoss(approx_f_sgd)
    srgd_loss = SquareLoss(approx_f_srgd)

    # Create optimizers
    sgd = SGD(loss_func=sgd_loss,
              approx_func=approx_f_sgd,
              gradient=grad,
              learning_rate=learning_rate)
   
    srgd = SRGD(loss_func=srgd_loss,
                approx_func=approx_f_srgd,
                gradient=grad,
                learning_rate=scaler * learning_rate,
                repeat_num=r)
    
    sgd.set_log_rate(100)
    srgd.set_log_rate(100)

    logger.info("Running SGD optimization")
    with Timer() as t:
        (n_iter, n_epochs, errors) = sgd.solve(
                data=features,
                labels=labels,
                num_epochs=n_epochs,
                batch_size=mini_batch_size)
    logger.info("Completed SGD optimization in {:,} iterations and in {:.5f} seconds".format(
        n_iter, t.interval))
    logger.info("SGD error: {}".format(sgd.get_error(np.concatenate(
        (features, labels.reshape(labels[n_train:].size,1)),axis=1))))

    logger.info("Running SRGD optimization")
    with Timer() as t:
        (n_iter, n_epochs, errors) = srgd.solve(
                data=features,
                labels=labels,
                num_epochs=n_epochs,
                batch_size=mini_batch_size)
    logger.info("Completed SRGD optimization in {:,} iterations and in {:.5f} seconds".format(
        n_iter, t.interval))
    logger.info("SRGD error: {}".format(srgd.get_error(np.concatenate(
        (features, labels.reshape(labels[n_train:].size,1)), axis=1))))


def main():
    """MAIN"""
    labels, features = load_data("data/covtype/covtype.libsvm.binary.scale")
    print("Loaded data")
    print("There are {} labels".format(len(labels)))
    print("Feature matrix has shape {}".format(features.shape))
    fit_least_squares(labels, features)
if __name__ == "__main__":
    main()
