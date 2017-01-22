import logging
import logging.config
import yaml
import os
import numpy as np
import datetime
from generate_charts import generate_chart
import function

# Setup logging
with open("logging.yaml", "r") as fd:
    config = yaml.safe_load(fd.read())

# Add timestamp to log file
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

def create_coefficients(n_dim, min_val=-100, max_val=101):
    """Returns a numpy array of size n_dim with randomly
    generated coefficients on the interval [min_val, max_val)
    """
    return np.random.randint(min_val, max_val, n_dim)

def get_gradient(diff, x_vals):
    """Gradient for linear function and square loss"""
    return diff * x_vals

def main():
    """Main function"""
    # array of problem sizes 
    dimension = 5
    n_data = dimension*5
    mini_batch_sizes = [1, 10, 20, 50]

    coefficients = create_coefficients(dimension)
    test_func = LinearFunction(coefficients)
    
    generate_chart()

if __name__ == "__main__":
    main()
