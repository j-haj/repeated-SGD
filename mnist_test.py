import logging
import logging.config
import datetime
import time
import yaml
import os
import numpy as np

# Setup logging
with open("logging.yaml", "r") as fd:
    config = yaml.safe_load(fd.read())

# Add timestamp to file
log_filename = config9"handlers"]["file"]["filename"]
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

def square_loss(y, y_hat):
    """Calculates the square loss for the given y values"""
    return 0.5 * (y - y_hat)**2


def gradient(x_vals, y):
    return -1 * (y - y_hat) * np.concatenate(x_vals, [1])

def sgd(approx_func, data):


def main():
    """Main function"""

    # Load MNIST training data

    # Build model scaffolding

    # Train model

    # Test model on test data
    pass

if __name__ == "__main__":
    main()
