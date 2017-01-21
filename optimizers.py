import numpy as np

class Optimizer:
    """Optimizer class is a base class for various optimizers and
    should not be instantiated on its own.
    """

    def __init__(self, func=None, gradient=None):
        """Must provide both a function and a gradient
        for evaluating y_hat as well as getting gradient
        values.
        """
        self.function = func
        self.gradient = gradient

    
class SGD(Optimizer):

    def __init__(self, func=None, gradient=None, learning_rate):
        super(SGD, self).__init__(func, gradient)
        self.learning_rate = learning_rate

    def update_weights(self, data, labels, weights):
        """Runs a single weight update and returns the updated weights"""

        pass

class ASSGD(Optimizer):
    """Accelerated Stochastic Gradient class - similar to SGD but uses
    and adaptive step size.
    """
    def __init__(self, func=None, gradient=None, threshold=0.01,
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


    def update_weights(self, data, labels, weights):
        """Runs a single weight update and returns the updated weights"""
    
    
def sgd(data, weights, func=None, gradient=None, learning_rate=0.0001):
    """Returns updated weights based on a fixed steplearning rate.
    

    Return: numpy array containing updated weights
    """
    
    # Need to make sure a gradient and func functions are passed in
    assert gradient is not None
    assert func is not None

    x_vals = data[:,:-1]
    y = data[:, -1]
    aggregate_grad = 0

    for i in range(y.size):
        # Get diff between true y and estimated y
        y_hat = func(x_vals[i,:], weights)
        diff = y_hat - y[i]

        # Get gradient
        aggregate_grad += gradient(diff, x_vals[i, :])

    aggregate_grad /= y.size

    weights = weights - learning_rate * aggregate_grad

def assgd(data,
          weights,
          func=None,
          gradient=None,
          learning_rate=0.001,
          threshold=0.001):
    """Accelerated stochastic subgradient method - see paper (Xi, Lin, Yang
    2016). 

    Return: numpy array containing updated weights
    """

    assert gradient is not None
    assert func is not None

    x_vals = data[:, :-1]
    y = data[:, -1]
    aggregate_grad = 0

