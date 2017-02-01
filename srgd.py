import csv
import logging
import logging.config
import yaml
import os
import numpy as np
import datetime
import time
from core.function.function import LinearFunction
from core.function.loss_function import SquareLoss
from core.optimizers.optimizers import SGD, ASSGD, SRGD
from sklearn import preprocessing

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

def write_results_to_file(results, filename):
    """Results file format:

    <optimizer> <dimension> <r> <batch_size> <num_iteration> <runtime> <error1> <error2>
    """
    with open(filename, 'w') as writer:
        # TODO: write header to file

        for key in results:
            
            results_list = results[key]
            for row in results_list:
                file_str = ("{optimizer} {dim} {r} {bs} {niter} {time} {e1} "
                "{e2}\n").format(optimizer=key,
                        dim=row["dimension"],
                        r=row["r"],
                        bs=row["batch_size"],
                        niter=row["n_iter"],
                        time=row["time"],
                        e1=row["error"][-1][1]["norm"],
                        e2=row["error"][-1][1]["max"])
                writer.write(file_str)


def create_coefficients(n_dim, min_val=-10, max_val=11):
    """Returns a numpy array of size n_dim with randomly
    generated coefficients on the interval [min_val, max_val)
    """
    return np.random.randint(min_val, max_val, n_dim)

def generate_labeled_data(n_data, func, x_min=-10, x_max=11):
    labels = np.zeros(n_data)
    data = np.zeros((n_data, func.parameters.size))
    for i in range(n_data):
        x_data = np.random.randint(x_min, x_max, func.parameters.size)
        label = func.evaluate(x_data)
        labels[i] = label
        data[i] = x_data
    return (data, labels)

def scale_data(data):
    """Normalizes data by applying the following transform:

        z = (x - mean)/std_dev

    Parameters:
        data: data to be normalized

    Return: normalized data
    """
    return preprocessing.scale(data)

def get_gradient(diff, x_vals):
    """Gradient for linear function and square loss"""
    return diff * x_vals


def run_short_experiment(dimension, learning_rate):
    """Runs experiment but on a much smaller scale"""

    # Create results dicitonary
    results = {"sgd": [], "assgd": [], "srgd": [], "assrgd": []}
    # Run tests
    num_tests = 5
    mini_batch_sizes = [1, 50, 100, 500, 1000]
    r_factors = [i for i in range(1,11)]

    for i in range(num_tests):
        logger.info("Test iteration {}".format(i))

        # Create coefficients
        data_size_factor = 1000
        coefficients = create_coefficients(dimension)
        test_func = LinearFunction(dim=dimension, parameters=coefficients)
        n_data = dimension * data_size_factor
        x_vals, labels = generate_labeled_data(n_data,
                                               test_func)
        logger.info("Parameters, test function, and data generated")

        # Iterate over mini batch sizes
        for batch_size in mini_batch_sizes:
            logger.info("Running mini_batch size {}".format(batch_size))
            # Iterate over repetition factors
            for r in r_factors:
                logger.info("Using r = {}".format(r))
                # Create approx functions
                logger.info("Creating approximate funtions")
                approx_f_sgd = LinearFunction(dim=dimension)
                approx_f_assgd = LinearFunction(dim=dimension)
                approx_f_srgd = LinearFunction(dim=dimension)
                approx_f_assrgd = LinearFunction(dim=dimension)
                logger.info("Approximate functions created")
                
                # Create loss functions
                sgd_loss = SquareLoss(approx_f_sgd)
                assgd_loss = SquareLoss(approx_f_assgd)
                srgd_loss = SquareLoss(approx_f_srgd)
                assrgd_loss = SquareLoss(approx_f_assrgd)

                # Create optimization instances
                sgd = SGD(loss_func=sgd_loss,
                          approx_func=approx_f_sgd,
                          gradient=get_gradient,
                          learning_rate=learning_rate)

                assgd = ASSGD(loss_func=assgd_loss,
                              approx_func=approx_f_assgd,
                              gradient=get_gradient,
                              threshold=0.001,
                              learning_rate=learning_rate)

                srgd = SRGD(loss_func=srgd_loss,
                            approx_func=approx_f_srgd,
                            gradient=get_gradient,
                            learning_rate=learning_rate,
                            repeat_num=r)

                assrgd = SRGD(loss_func=assrgd_loss,
                              approx_func=approx_f_assrgd,
                              gradient=get_gradient,
                              threshold=0.001,
                              learning_rate=learning_rate,
                              repeat_num=r)
                optimizers = {"sgd": sgd, "assgd": assgd, "srgd": srgd, "assrgd": assrgd}
                logger.info("Optimizers successfully created")

                # Solve problem and collect errors
                for key in results:
                    o = optimizers[key]
                    logger.info("Running optimization on {}".format(key))
                    with Timer() as t:
                        (n_iter, n_epoch, errors) = o.solve(
                            data=x_vals,
                            labels=labels,
                            num_epochs=n_data//batch_size,
                            batch_size=batch_size)
                    test_res = {"n_iter": n_iter,
                                "n_epoch": n_epoch,
                                "error": errors,
                                "time": t.interval,
                                "batch_size": batch_size,
                                "dimension": dimension,
                                "r": r}
                    logger.info("Completed {} optimization in {:.5f} seconds".format(key, t.interval))
                    logger.info("{} error: {}".format(key,
                        optimizers[key].get_error(np.concatenate((x_vals, labels.reshape(labels.size, 1)),
                                                                 axis=1))))
                    results[key].append(test_res)
    return results

def run_experiment():
    """Run experiment to collect results for paper"""
    # Create results dictionary
    results = {"sgd": [], "assgd": [], "srgd": [], "assrgd": []}

    # Create step size array and dimension step array
    dimension_steps = [50, 100, 500, 1000, 10000]
    mini_batch_sizes = []
    data_size_factor = 1000 # Number of data points based on problem dimension
    learning_rate = 0.0001

    mini_batch_sizes = [1, 50, 100, 500, 1000]
    learning_rates = [0.0001, 0.0001, 0.001, 0.01, 0.01]

    logger.info("Mini batch array: {}".format(mini_batch_sizes))

    r_factors = [i for i in range(1, 11)]

    # Run tests
    num_tests = 10
    for i in range(num_tests):
        logger.info("Test iteration {}".format(i))

        # Iterate over various problem sizes
        for idx, dim in enumerate(dimension_steps):

            learning_rate = learning_rates[idx]

            # Create coefficients
            coefficients = create_coefficients(dim)
            test_func = LinearFunction(dim=dim, parameters=coefficients)
            n_data = dim * data_size_factor
            x_vals, labels = generate_labeled_data(n_data,
                                                   test_func)
            logger.info("Parameters, test function, and data generated")

            # Iterate over mini batch sizes
            for batch_size in mini_batch_sizes:
                logger.info("Running mini_batch size {}".format(batch_size))
                # Iterate over repetition factors
                for r in r_factors:
                    logger.info("Using r = {}".format(r))
                    # Create approx functions
                    logger.info("Creating approximate funtions")
                    approx_f_sgd = LinearFunction(dim=dim)
                    approx_f_assgd = LinearFunction(dim=dim)
                    approx_f_srgd = LinearFunction(dim=dim)
                    approx_f_assrgd = LinearFunction(dim=dim)
                    logger.info("Approximate functions created")

                    # Create optimization instances
                    sgd = SGD(func=test_func,
                              approx_func=approx_f_sgd,
                              gradient=get_gradient,
                              learning_rate=learning_rate)

                    assgd = ASSGD(func=test_func,
                                  approx_func=approx_f_assgd,
                                  gradient=get_gradient,
                                  threshold=0.001,
                                  learning_rate=learning_rate)

                    srgd = SRGD(func=test_func,
                                approx_func=approx_f_srgd,
                                gradient=get_gradient,
                                learning_rate=learning_rate,
                                repeat_num=r)

                    assrgd = SRGD(func=test_func,
                                  approx_func=approx_f_assrgd,
                                  gradient=get_gradient,
                                  threshold=0.001,
                                  learning_rate=learning_rate,
                                  repeat_num=r)
                    optimizers = {"sgd": sgd, "assgd": assgd, "srgd": srgd, "assrgd": assrgd}
                    logger.info("Optimizers successfully created")

                    # Solve problem and collect errors
                    for key in results:
                        o = optimizers[key]
                        logger.info("Running optimization on {}".format(key))
                        with Timer() as t:
                            (n_iter, n_epoch, errors) = o.solve(
                                data=x_vals,
                                labels=labels,
                                num_epochs=n_data//batch_size,
                                batch_size=batch_size)
                        test_res = {"n_iter": n_iter,
                                    "n_epoch": n_epoch,
                                    "error": errors,
                                    "time": t.interval,
                                    "batch_size": batch_size,
                                    "dimension": dim,
                                    "r": r}
                        logger.info("Completed {} optimization in {:.5f} seconds".format(key, t.interval))
                        logger.info("{} results: {}".format(key, test_res))
                        results[key].append(test_res)
    return results

def main():
    """Main function"""
    # array of problem sizes 
    dimension = 50
    n_data = dimension*1000
    mini_batch_sizes = [1, 10, 20, 50]
    logger.info("Test dimension: {}".format(dimension))
    logger.info("Number of data points: {}".format(n_data))

    # Create coefficients and true func logger.info("Creating coefficients and true function")
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
    learning_rate = 0.0001 
    batch_size = 10
    repeat_num = 3
    sgd = SGD(func=test_func,
            approx_func=approx_f_sgd,
            gradient=get_gradient,
            learning_rate=learning_rate)
    assgd = ASSGD(func=test_func,
            approx_func=approx_f_assgd,
            gradient=get_gradient,
            threshold=0.001,
            learning_rate=learning_rate)
    srgd = SRGD(func=test_func,
            approx_func=approx_f_srgd,
            gradient=get_gradient,
            learning_rate=learning_rate,
            repeat_num=repeat_num)
    assrgd = SRGD(func=test_func,
            approx_func=approx_f_assrgd,
            gradient=get_gradient,
            threshold=0.001,
            learning_rate=learning_rate,
            repeat_num=repeat_num)

    sgd.set_log_rate(50)
    assgd.set_log_rate(50)
    srgd.set_log_rate(50)
    assrgd.set_log_rate(50)

    logger.info("Initialization complete - beginning tests...")
    with Timer() as t: 
        (s_iter, s_epoch, s_err) = sgd.solve(data=x_vals,
                                   labels=labels,
                                   num_epochs=n_data//batch_size,
                                   batch_size=batch_size)
    logger.info(("SGD solved coefficients in {:,} steps, {:.5f} seconds, with "
        "{} error").format(s_iter,
                                    t.interval,
                                    s_err))
    with Timer() as t:
        assgd_iter, assgd_epch, assgd_err = assgd.solve(data=x_vals,
                                       labels=labels,
                                       num_epochs=n_data//batch_size,
                                       batch_size=batch_size)
    logger.info(("ASSGD solved coefficients in {:,} steps, {:.5f} seconds, with "
        "{} error").format(assgd_iter,
                                    t.interval,
                                    assgd_err))
    with Timer() as t:
        srgd_iter, srgd_epch, srgd_err = srgd.solve(data=x_vals,
                                     labels=labels,
                                     num_epochs=n_data//batch_size,
                                     batch_size=batch_size)
    logger.info(("SRGD solved coefficients in {:,} steps, {:.5f} seconds, with "
        "{} error").format(srgd_iter,
                                    t.interval,
                                    srgd_err))
    with Timer() as t:
        assrgd_iter, assrgd_epch, assrgd_err = assrgd.solve(data=x_vals,
                                         labels=labels,
                                         num_epochs=n_data//batch_size,
                                         batch_size=batch_size)
    logger.info(("ASSGRD solved coefficients in {:,} steps, {:.5f} seconds, with "
        "{} error").format(assrgd_iter,
                                    t.interval,
                                    assrgd_err))

if __name__ == "__main__":
    
    #main()
    logger.info("Running experiment")
    #result = run_experiment()
    results = run_short_experiment(10, .000001)
    logger.info("Experiment Complete!")
    results_filename = "experiment_results.yaml"
    write_results_to_file(results, "results_1000_01.csv")
    # Write results to file
    #logger.info("Writing results to file")
    #with open(results_filename, 'w') as out_file:
    #    yaml.dump(results, out_file, default_flow_style=True)
    #logger.info("Finished.")
