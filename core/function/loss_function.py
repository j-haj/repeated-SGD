from core.function.function import LinearFunction
import numpy as np

class SquareLoss():

    def __init__(self, approx_func):
        """Initializes approximate function for loss evaluation.
        """
        self.approx_func = approx_func

    def evaluate(self, x_vals, y):
        """Compares the square loss between the estimate y_hat and the
        true label y

        Return: 1/2(y - y_hat)^2
        """
        if y.size > 1:
            avg_err = 0
            for i in range(y.size):
                avg_err += (y[i] - self.approx_func.evaluate(x_vals[i, :]))**2
            return 0.5 * avg_err / y.size
        return 0.5*(y - self.approx_func.evaluate(x_vals))**2

    def gradient(self, x_vals, y):
        """Returns the gradient of the loss function"""
        return -np.dot((y - self.approx_func.evaluate(x_vals)), self.approx_func.gradient(x_vals))
