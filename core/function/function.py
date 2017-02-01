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

    def gradient(self, x_vals):
        """Interface method - this hsould be implemented by inherited class"""
        logger.error("Failed to implement virtual method `gradient`")
        raise NotImplementedError("Must implement this method in inherited class")

    def update_parameters(self, parameters):
        """Interface method - this should be implemented by inherited class"""
        logger.error("Failed to implement virtual method `update_parameters`")
        raise NotImplementedError

class SingleLayerNetwork(Function):
    """Class of neural networks with a single hidden layer"""

    def __init__(self, dim, num_neurons, p_min=0, p_max=1):
        """

        Parameters:
            dim: input dimension
            num_neurons: number of neurons in the layer, this means there are
                         (dim + 1)*num_neurons parameters in the model
            parameters: starting parameter values
            p_min: min value for randomly initialized parameters
            p_max: max value for randomly initialized parameters
        """
        self.parameters = np.empty([num_neurons, dim + 1])
        for i in range(num_neurons):
            self.parameters[i] = np.random.randin(p_min, p_max, (dim + 1))
        super(SingleLayerNetwork).__init__(self.parameters)
        self.num_neurons = num_neurons
        self.dim = dim
        
        
    def evaluate(self, x_vals):
        """Return the output of the single layer"""
        
        # extend x_vals by 1 for bias
        x_vals = np.concatenate((x_vals, np.array([1])))

        # the last parameter in each row is the bias
        return np.dot(self.parameters, x_vals)

    def gradient(self, x_vals):
        """Gradient function. Note that the last parameter for each neuron
        is a bias, so it's derivative is 1.
        """
        rows = x_vals.size[0]
        ones = np.ones(rows).reshape((rows, 1))
        return np.concatenate((x_vals, ones), axis=1) 

    def update_parameters(self, update_val):
        """Updates parameter matrix"""
        logger.debug("Old parameters: {}".format(self.parameters))
        self.parameters = np.subtract(self.parameters, update_val)
        logger.debug("New parameters: {}".format(self.parameters))

class LinearFunction(Function):
    """Class of functions that take the form
        y = a_1 * x_1 + a_2 * x_2 + ... + a_n * x_n
    """

    def __init__(self, dim, parameters=None, p_min=0, p_max=1):
        """Initializer

        Parameters:

            dim: dimension of independent var
            parameters: parameters of function (a_i)
        """
        if parameters is None:
            logger.info("Paramters is None - generating random paramters between [0, 1]")
            parameters = np.random.randint(p_min, p_max, dim)
        super(LinearFunction, self).__init__(parameters)
        self.dim = dim

    def evaluate(self, x_vals):
        """Return inner product of x_vals with parameters"""
        if x_vals.size != self.parameters.size:
            logger.info("x_vals: {}".format(x_vals))
            logger.info("self.parameters: {}".format(self.parameters))
        return np.dot(self.parameters, x_vals)

    def gradient(self, x_vals):
        """Returns the gradient of a linear function"""
        return x_vals

    def update_parameters(self, update_val):
        """Updates parameter array"""
        logger.debug("Old parameters: {}".format(self.parameters))
        self.parameters = np.subtract(self.parameters, update_val)
        logger.debug("New parameters: {}".format(self.parameters))
