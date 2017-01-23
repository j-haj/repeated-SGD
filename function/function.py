import numpy as np
import logging

logger = logging.getLogger()

class Function:
    """Abstract class of functions"""

    def __init__(self, parameters=None):
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

    def __init__(self, dim, parameters=None):
        """Initializer

        Parameters:

            dim: dimension of independent var
            parameters: parameters of function (a_i)
        """
        if parameters is None:
            logger.info("Paramters is None - generating random paramters between [0, 1]")
            parameters = np.random.randint(0, 1, dim)
        super(LinearFunction, self).__init__(parameters)
        self.dim = dim

    def evaluate(self, x_vals):
        """Return inner product of x_vals with parameters"""
        return np.dot(self.parameters, x_vals)
