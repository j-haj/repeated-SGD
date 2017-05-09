import numpy as np
import logging
from ..function.function import LinearFunction, Function
import time

logger = logging.getLogger()

class Timer:

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

class Optimizer:
    """Optimizer class is a base class for various optimizers and
    should not be instantiated on its own.
    """

    def __init__(self,
                 loss_func=None,
                 approx_func=None,
                 gradient=None,
                 learning_rate=0.000001,
                 log_thresh=1e-3):
        """Must provide both a function and a gradient
        for evaluating y_hat as well as getting gradient
        values.
        """
        self.loss_func = loss_func
        self.approx_func = approx_func
        self.gradient = gradient
        self._learning_rate = learning_rate
        self.training_error = []
        self.tail_error = []
        self.errors = []
        self._max_iter = 100000
        self.log_rate = 0
        self.log_thresh = log_thresh
        self.time = 0

    def get_error(self, data):
        """Returns the current error over the entire dataset using
        the registered error function
        """
        x_vals = data[:, :-1]
        y = data[:, -1]

        total_error = 0
        for i in range(y.size):
            total_error += self.loss_func.evaluate(x_vals[i, :], y[i])
        
        return total_error / y.size 

    def update_weights(self, data):
        """Interface method - should be implemented in subclasses"""
        logger.error("Attempted to instantiate abstract Optimizer class")
        raise NotImplementedError("Must implement this method in inherited class")
    
    def set_log_rate(self, n):
        """Set the interval at which status data is logged.
        Default value is 0 - which means no log data for
        intermediate results.
        """
        self.log_rate = n

    def solve(self,
              data,
              labels,
              num_epochs=None,
              batch_size=1,
              early_stop=1e-6):
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

        old_weights = np.array([5 for _ in range(data[0, :].size)])
        logger.info("Using mini-batch of size {}".format(batch_size))
        idx = 0
        epoch_idx = 0
        with Timer() as t:
            # Each epoch should iterate over the entire dataset
            t_start = time.clock()
            for epoch_num in range(num_epochs):
                n_data = data.shape[0]
                n_sub_epochs = n_data//batch_size
                for sub_epoch in range(n_sub_epochs):
                    logger.debug("Sub-epoch {}".format(sub_epoch))
                    if idx > self._max_iter or np.allclose(old_weights,
                                                           self.approx_func.parameters,
                                                           early_stop):
                        logger.debug("np.allclose = {}".format(np.allclose(old_weights,
                                                           self.approx_func.parameters,
                                                           early_stop)))
                        if idx > self._max_iter:
                            logger.info("[iteration {}]Stopping early: max number of iterations".format(
                                idx))
                        else:
                            logger.info("[iteration {}]Stopping early: tail convergence".format(
                                idx))
                        break

                    # Get batch
                    length = data.shape[0]
                    batch_indices = np.random.choice(length,
                                                     batch_size)

                    # Need to reshape the batch to properly concatenate
                    mini_batch = np.concatenate((data, labels.reshape(labels.size, 1)),
                                                axis=1)[batch_indices]
                    # Run weight update
                    if idx > 0:
                        tail_diff = np.subtract(old_weights,
                                self.approx_func.parameters)
                        train_diff = self.loss_func.evaluate(data, labels)                    
                        max_tail_err = np.max(np.fabs(tail_diff))
                        norm_tail_err = np.linalg.norm(tail_diff)
                        max_train_err = np.max(np.fabs(train_diff))
                        norm_train_err = np.linalg.norm(train_diff)

                        self.tail_error.append((max_tail_err, norm_tail_err))
                        self.training_error.append((max_train_err, norm_train_err))
                        if self.log_rate > 0 and idx % self.log_rate == 0 or\
                            norm_train_err < self.log_thresh:
                            if norm_train_err < self.log_thresh:
                                self.log_thresh /= 10

                            logger.info("[{}] Tail error - max:  {:.4f}".format(
                                idx, max_tail_err))
                            logger.info("[{}] Tail error - norm: {:.4f}".format(
                                idx, norm_tail_err))
                            logger.info("[{}] Train error - max:  {:.4f}".format(
                                idx, max_train_err))
                            logger.info("[{}] Train error - norm: {:.4f}".format(
                                idx, norm_train_err))

                    old_weights = self.approx_func.parameters
                    idx += self.update_weights(mini_batch)
                    epoch_idx += 1
                    current_err = self.get_error(mini_batch)
                    logger.debug("Iteration: {}\t Error: {}".format(idx, current_err))
                    self.errors.append((idx, current_err, time.clock() - t_start))
                logger.info("Epoch: {}\tError: {}".format(epoch_num,
                    self.errors[-1]))

        self.time = t.interval
        return (idx, epoch_idx, self.errors)

class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer."""

    def __init__(self,
                 loss_func=None,
                 approx_func=None,
                 gradient=None,
                 learning_rate=0.0001):
        super(SGD, self).__init__(loss_func, approx_func, gradient, learning_rate)

    def update_weights(self, data):
        """Runs a single weight update and returns the updated weights"""
        # Need to make sure a gradient and func functions are passed in

        x_vals = data[:, :-1]
        y = data[:, -1]
        aggregate_grad = 0

        for i in range(y.size):
            # Get gradient
            aggregate_grad = np.add(aggregate_grad,
                    self.loss_func.gradient(x_vals[i, :], y[i]))

        aggregate_grad = np.divide(aggregate_grad, y.size)
        logger.debug("SGD gradient: {}".format(aggregate_grad))
        self.approx_func.update_parameters(self._learning_rate * aggregate_grad)
        return 1

class ASSGD(Optimizer):
    """Accelerated Stochastic Gradient class - similar to SGD but uses
    and adaptive step size.
    """
    def __init__(self,
                 loss_func=None,
                 approx_func=None,
                 gradient=None,
                 learning_rate=0.0001,
                 threshold=0.01,
                 step_factor=0.5,
                 threshold_step=0.1):
        """Initializer

        Parameters:
            threshold: used to determine when the step size is decreased
            step_factor: factor used to decrease the step size, applied
                         geometrically
            func: function approximating the target function
            gradient: function that returns the gradient
        """
        super(ASSGD, self).__init__(loss_func, approx_func, gradient, learning_rate)
        self._threshold = threshold
        self._step_factor = step_factor
        self._threshold_step = threshold_step

    def update_weights(self, data):
        """Runs a single weight update and returns the updated weights"""

        x_vals = data[:, :-1]
        y = data[:, -1]
        aggregate_grad = 0
        old_weights = self.approx_func.parameters
        
        for i in range(y.size):
            aggregate_grad = np.add(aggregate_grad,
                    self.loss_func.gradient(x_vals[i, :], y[i]))

        aggregate_grad = np.divide(aggregate_grad, y.size)
        logger.debug("ASSGD gradient: {}".format(aggregate_grad))
        self.approx_func.update_parameters(self._learning_rate * aggregate_grad)

        # Check diff and update learning rate if necessary
        if np.allclose(self.approx_func.parameters,
                       old_weights, self._threshold):
            self._learning_rate *= self._step_factor
            self._threshold *= self._threshold_step

            logger.debug(("Difference in y_hat values below threshold - "
                    "decreasing the step-size by a factor of {} to {}").format(
                        self._learning_rate, self._step_factor))
        return 1

class SRGD(Optimizer):
    """Stochastic Repeated Gradient Descent using adaptive step-size"""

    def __init__(self,
                 loss_func=None,
                 approx_func=None,
                 gradient=None,
                 threshold=0,
                 threshold_step=0.1,
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
        super(SRGD, self).__init__(loss_func, approx_func, gradient, learning_rate*2)
        self._threshold = threshold
        self._step_factor = step_factor
        self._threshold_step = threshold_step
        self._repeat_num = repeat_num

    def update_weights(self, data):
        """ Runs SRGD update on the weight self.repeat_num of times"""

        x_vals = data[:, :-1]
        y = data[:, -1]
        old_weights = self.approx_func.parameters

        for it in range(self._repeat_num):
            aggregate_grad = 0
            for i in range(y.size):
                aggregate_grad = np.add(aggregate_grad,
                        self.loss_func.gradient(x_vals[i, :], y[i]))

            aggregate_grad = np.divide(aggregate_grad, y.size)
            logger.debug("SRGD gradient: {}".format(aggregate_grad))
            self.approx_func.update_parameters(aggregate_grad  * self._learning_rate)

        # Check diff and update learning rate if necessary
        if self._threshold > 0 and \
           np.allclose(self.approx_func.parameters,
                       old_weights, self._threshold):
            self._learning_rate *= self._step_factor
            self._threshold *= self._threshold_step

            logger.debug(("Difference in y_hat values below threshold - "
                    "decreasing the step-size by a factor of {} to {}").format(
                        self._learning_rate, self._step_factor))
        return self._repeat_num
