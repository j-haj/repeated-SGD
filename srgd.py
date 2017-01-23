import logging
import logging.config
import yaml
import os
import numpy as np
import datetime
from core.function.function import LinearFunction
from core.optimizers.optimizers import SGD, ASSGD, SRGD

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

def generate_labeled_data(n_data, func, x_min=-1000, x_max=1000):
    labels = np.zeros(n_data)
    data = np.zeros((n_data, func.parameters.size))
    for i in range(n_data):
        x_data = np.random.randint(x_min, x_max, func.parameters.size)
        label = func.evaluate(x_data)
        labels[i] = label
        data[i] = x_data
    return (data, labels)

def get_gradient(diff, x_vals):
    """Gradient for linear function and square loss"""
    logger.debug("diff: {} - x_vals: {}".format(diff, x_vals))
    return diff * x_vals

def main():
    """Main function"""
    # array of problem sizes 
    dimension = 5
    n_data = dimension*1000
    mini_batch_sizes = [1, 10, 20, 50]
    logger.info("Test dimension: {}".format(dimension))
    logger.info("Number of data points: {}".format(n_data))

    # Create coefficients and true func
    logger.info("Creating coefficients and true function")
    coefficients = create_coefficients(dimension)
    logger.info("Created coefficients: {}".format(coefficients))
    test_func = LinearFunction(dim=dimension, parameters=coefficients)
    x_vals, labels = generate_labeled_data(n_data,
                                           test_func)
    
    # Approx functions
    logger.info("Creating approximate linear function objects")
    approx_f_sgd = LinearFunction(dim=dimension)
    approx_f_assgd = LinearFunction(dim=dimension)
    approx_f_srgd = LinearFunction(dim=dimension)
    approx_f_assrgd = LinearFunction(dim=dimension)

    # Initialize optimizers
    logger.info("Initializing optimizer objects")
    sgd = SGD(func=test_func,
            approx_func=approx_f_sgd,
            gradient=get_gradient,
            learning_rate=0.000001)
    assgd = ASSGD(func=test_func,
            approx_func=approx_f_assgd,
            gradient=get_gradient,
            learning_rate=0.000001)
    srgd = SRGD(func=test_func,
            approx_func=approx_f_srgd,
            gradient=get_gradient,
            learning_rate=0.00001)
    assrgd = SRGD(func=test_func,
            approx_func=approx_f_assrgd,
            gradient=get_gradient,
            threshold=0.01,
            learning_rate=0.00001)

    sgd.set_log_rate(500)
    assgd.set_log_rate(500)
    srgd.set_log_rate(500)
    assrgd.set_log_rate(500)

    logger.info("Initialization complete - beginning tests...")
    
    sgd_step_count = sgd.solve(x_vals, labels, n_data)
    logger.info("SGD solved coefficients in {} steps: {}".format(sgd_step_count,
                                                                 sgd.approx_func.parameters))
    assgd_step_count = assgd.solve(x_vals, labels, n_data)
    logger.info("ASSGD solved coefficients in {} steps: {}".format(assgd_step_count,
                                                                   assgd.approx_func.parameters))
    srgd_step_count = srgd.solve(x_vals, labels, n_data)
    logger.info("SRGD solved coefficients in {} steps: {}".format(srgd_step_count,
                                                                  srgd.approx_func.parameters))
    assrgd_step_count = assrgd.solve(x_vals, labels, n_data)
    logger.info("ASSGRD solved coefficients in {} steps: {}".format(assrgd_step_count,
                                                        assrgd.approx_func.parameters))

if __name__ == "__main__":
    main()
