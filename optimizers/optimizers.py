import numpy as np
import logging
from function import *

logger = logging.getLogger()

class Optimizer:
    """Optimizer class is a base class for various optimizers and
    should not be instantiated on its own.
    """

    def __init__(self,
                 func=None,
                 approx_func=None,
                 gradient=None,
                 learning_rate=0.0001):
        """Must provide both a function and a gradient
        for evaluating y_hat as well as getting gradient
        values.
        """
        self._true_coeff = func.parameters
        self.function = func
        self.approx_func = approx_func
        self.gradient = gradient
        self._learning_rate = learning_rate
        self.error_tracker = []

        self._max_iter = 100000

    def update_weights(self, data):
        """Interface method - should be implemented in subclasses"""
        logger.error("Attempted to instantiate abstract Optimizer class")
        raise NotImplementedError("Must implement this method in inherited class")

    def solve(self,
              data,
              labels,
              num_epochs=None,
              batch_size=1,
              early_stop=0.000001):
        """Solves for the weights that minimize there loss functoin

        Parameters:
            data: x data
            labels: y data
            num_epochs: number of epochs to perform. If None,
                        the number of epochs is chosen based on the
                        batch size
            batch_size: size of the mini batch in each epoch
            early_stop: threshold at which to stop training
        """

        old_weights = np.array([100 for _ in range(data[0, :].size)])
        logger.info("Using mini-batch of size {}".format(data.shape[0]))
        idx = 0
        for epoch_num in range(num_epochs):

            if idx > self._max_iter or np.allclose(old_weights, self.approx_func.parameters):
                if idx > self._max_iter:
                    logger.info("Stopping early due to exceeds max number of iterations")
                else:
                    logger.info("Stopping early due to slowed convergence")
                break

            # Get batch
            length = data.shape[0]
            batch_indices = np.random.choice(length,
                                             batch_size)
            mini_batch = np.concatenate((data, labels.reshape(labels.size, 1)),
                                        axis=1)[batch_indices]

            # Run weight update
            old_weights = self.approx_func.parameters
            self.update_weights(mini_batch)

class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""

    def __init__(self,
                 func=None,
                 approx_func=None,
                 gradient=None,
                 learning_rate=0.0001):
        super(SGD, self).__init__(func, approx_func, gradient, learning_rate)

    def update_weights(self, data):
        """Runs a single weight update and returns the updated weights"""
        # Need to make sure a gradient and func functions are passed in

        x_vals = data[:, :-1]
        y = data[:, -1]
        aggregate_grad = 0

        for i in range(y.size):
            # Get diff between true y and estimated y
            y_hat = self.approx_func.evaluate(x_vals[i, :])
            diff = y_hat - y[i]

            # Get gradient
            aggregate_grad += self.gradient(diff, x_vals[i, :])

        aggregate_grad /= y.size

        self.approx_func.parameters = np.subtract(self.approx_func.parameters,
                                                  self._learning_rate * aggregate_grad)

class ASSGD(Optimizer):
    """Accelerated Stochastic Gradient class - similar to SGD but uses
    and adaptive step size.
    """
    def __init__(self,
                 func=None,
                 approx_func=None,
                 gradient=None,
                 threshold=0.01,
                 step_factor=0.5):
        """Initializer

        Parameters:
            threshold: used to determine when the step size is decreased
            step_factor: factor used to decrease the step size, applied
                         geometrically
            func: function approximating the target function
            gradient: function that returns the gradient
        """
        super(ASSGD, self).__init__(func, approx_func, gradient)
        self._threshold = threshold
        self._step_factor = step_factor

    def update_weights(self, data):
        """Runs a single weight update and returns the updated weights"""

        x_vals = data[:, :-1]
        y = data[:, -1]
        aggregate_grad = 0

        for i in range(y.size):
            y_hat = self.approx_func.evaluate(x_vals[i, :])
            diff = y_hat - y[i]

            # Check diff and update learning rate if necessary
            if abs(diff) < self._threshold:
                self._learning_rate *= self._step_factor
                logger.info(("Difference in y_hat values below threshold - "
                        "decreasing the step-size by a factor of {} to {}").format(
                            self._learning_rate, self._step_factor))

            aggregate_grad += self.gradient(diff, x_vals[i, :])

        aggregate_grad /= y.size

        self.approx_func.parameters = np.subtract(self.approx_func.parameters,
                                                  self._learning_rate * aggregate_grad)

class SRGD(Optimizer):
    """Stochastic Repeated Gradient Descent using adaptive step-size"""

    def __init__(self,
                 func=None,
                 approx_func=None,
                 gradient=None,
                 threshold=0,
                 step_factor=0.5,
                 learning_rate=0.001,
                 repeat_num=10):
        """Initializer for SRGD. By default, adaptive step size is turned
        off (threshold=0), thus by default SRGD uses a fixed step size

        Parameters:

            func: function being optimized
            gradient: gradient function used by optimizer
            threshold: threshold value used to adjust step size (default of 0
                       means step size is fixed)
            step_factor: scale factor used to geometrically decrease the step size
            learning_rate: starting learning rate (step size), if threshold > 0,
                           this value will geometrically decrease over time
            repeat_num: number of iterations the weights are updated each time
        """
        super(SRGD, self).__init__(func, gradient, learning_rate)
        self._threshold = threshold
        self._step_factor = step_factor
        self._repeat_num = repeat_num

    def update_weights(self, data):
        """ Runs SRGD update on the weight self.repeat_num of times"""

        x_vals = data[:, :-1]
        y = data[:, -1]
        aggregate_grad = 0

        for _ in range(self._repeat_num):
            for i in range(y.size):
                y_hat = self.approx_func.evaluate(x_vals[i, :])
                diff = y_hat - y[i]

                # Check diff and update learning rate if necessary
                if abs(diff) < self._threshold:
                    self._learning_rate *= self._step_factor
                    logger.info(("Difference in y_hat values below threshold - "
                            "decreasing the step-size by a factor of {} to {}").format(
                                self._learning_rate, self._step_factor))

                aggregate_grad += self.gradient(diff, x_vals[i, :])

            aggregate_grad /= y.size

            self.approx_func.parameters = np.subtract(self.approx_func.parameters,
                                                      aggregate_grad * self._learning_rate)
