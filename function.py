import numpy as np
import logging

logger = logging.getLogger()

class Function:
    """Abstract class of functions"""

    def __init__(self, parameters):
        """Initializer

        Parameters:
            parameters: parameters that parameterize the function
        """
        self.parameters = parameters

    def evaluate(self, x_vals):
        """Inteface method - this should be implemented by inherited class"""
        logger.error("Failed to implement virtual method evaluate")
        raise NotImplementedError("Must implement this method in inherited class")

class LinearFunction(Function):
    """Class of functions that take the form
        y = a_1 * x_1 + a_2 * x_2 + ... + a_n * x_n
    """

    def evaluate(self, x_vals):
        """Return inner product of x_vals with parameters"""
        return np.dot(self.__parameters, x_vals)
