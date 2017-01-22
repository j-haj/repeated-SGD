import numpy as np
import logging

logger = logging.getLogger(__name__)

class Optimizer:
    """Optimizer class is a base class for various optimizers and
    should not be instantiated on its own.
    """

    def __init__(self,
                 func=None,
                 gradient=None,
                 learning_rate=0.0001):
        """Must provide both a function and a gradient
        for evaluating y_hat as well as getting gradient
        values.
        """
        self.function = func
        self.gradient = gradient
        self._learning_rate = learning_rate
        self.weight_tracker = []

        self.weights = np.array([])
        self._max_iter = 100000

    def update_weights(self):
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

        for epoch_num in range(num_epochs):

            if idx > self._max_iter or np.allclose(old_weights, self.weights):
                break

            # Get batch
            length = data.shape[0]
            batch_indices = np.random.choice(length,
                                             batch_size)
            mini_batch = np.concatenate((data, labels.reshape(labels.size, 1)),
                                        axis=1)[batch_indices]

            # Run weight update
            old_weights = self.weights
            self.update_weights(mini_batch)

class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""

    def __init__(self,
                 func=None,
                 gradient=None,
                 learning_rate=0.0001):
        super(SGD, self).__init__(func, gradient, learning_rate)

    def update_weights(self, weights, data):
        """Runs a single weight update and returns the updated weights"""
        # Need to make sure a gradient and func functions are passed in

        x_vals = data[:,:-1]
        y = data[:, -1]
        aggregate_grad = 0

        for i in range(y.size):
            # Get diff between true y and estimated y
            y_hat = self.function(x_vals[i, :], weights)
            diff = y_hat - y[i]

            # Get gradient
            aggregate_grad += self.gradient(diff, x_vals[i, :])

        aggregate_grad /= y.size

        weights -= self._learning_rate * aggregate_grad

class ASSGD(Optimizer):
    """Accelerated Stochastic Gradient class - similar to SGD but uses
    and adaptive step size.
    """
    def __init__(self,
                 func=None,
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
        super(ASSGD, self).__init__(func, gradient)
        self._threshold = threshold
        self._step_factor = step_factor

    def update_weights(self, weights, data):
        """Runs a single weight update and returns the updated weights"""

        x_vals = data[:, :-1]
        y = data[:, -1]
        aggregate_grad = 0

        for i in range(y.size):
            y_hat = self.function(x_vals[i, :], weights)
            diff = y_hat - y[i]

            # Check diff and update learning rate if necessary
            if abs(diff) < self._threshold:
                self._learning_rate *= self._step_factor

            aggregate_grad += self.gradient(diff, x_vals[i, :])

        aggregate_grad /= y.size

        weights -= self._learning_rate * aggregate_grad

        return weights
