import numpy as np
import logging
import time

class Timer:

    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

def create_coefficients(n_dim, l_bound=-100, u_bound=100):
    """Returns a Numpy array of size `n_dim` with randomly generated
    coefficients on the interval [l_bound, u_bound]
    """
    return np.random.random_integers(l_bound, u_bound, n_dim)

def evaluate_function(x_vals, coefficients):
    """Evaluates function parameterized by `coefficients` for input
    vector `x_vals`

    Parameters:

        x_vals: Numpy array
        coefficients: Numpy array

    """
    return np.dot(x_vals, coefficients)

def get_gradient(diff, x_vals):
    """Returns the gradient vector
    """
    return diff * x_vals

def generate_labeled_data(n_data, coefficients, x_bounds=(-1000, 1000)):
    """Generates `n_data` data points using the given
    `coefficients`. Returns a tuple containing (data, labels)
    """
    labels = np.zeros(n_data)
    data = np.zeros((n_data, coefficients.size))
    for i in range(n_data):
        x_data = np.random.random_integers(x_bounds[0],
                                           x_bounds[1],
                                           coefficients.size)
        label = evaluate_function(x_data, coefficients)
        labels[i] = label
        data[i] = x_data
    return (data, labels)

def repeated_epoch_sgd(data,
                      labels,
                      coefficients,
                      num_epochs=None,
                      repeat_num=2,
                      batch_size=1,
                      optimizer=None,
                      learning_rate=0.00001,
                      threshold=.00001,
                      max_iter=1000,
                      log_rate=0):
    """Runs sgd where parameters are updated once per epoch.
    The default batch size is 1, which yields standard online
    sgd."""

    assert optimizer is not None

    # Initialize weights
    weights = np.array([np.random.rand() for _ in range(coefficients.size)])
    old_weights = np.array([100 for _ in range(coefficients.size)])

    logging.debug("Initialized weights")

    if labels.size % batch_size != 0:
        batch_size = labels.size / (labels.size // batch_size)

    if num_epochs is None:
        num_epochs = labels.size // batch_size

    weight_hist = []
    idx = 1
    epochs_seen = 1
    num_data_pts_seen = 0
    for epoch_num in range(num_epochs):
        if idx > max_iter or np.allclose(old_weights, weights):
            break

        # get batch
        length = data.shape[0]
        batch_indices = np.random.choice(length,
                                         batch_size)

        mini_batch = np.concatenate((data, labels.reshape(labels.size, 1)),
                                    axis=1)[batch_indices]
        # run optimization on mini-batch
        for _ in range(repeat_num):
            old_weights = weights
            weights = optimizer(mini_batch, weights, learning_rate=learning_rate/repeat_num)
            weight_hist.append(weights)
            idx += 1

            if log_rate is not 0 and idx % log_rate == 0:
                print("[{}]: (repeated_epoch)norm_diff -> {}".format(idx,
                    np.linalg.norm(weights - old_weights)))

        epochs_seen += 1
        num_data_pts_seen += length

    return (idx, epochs_seen, num_data_pts_seen, np.array(weights))

def epoch_based_sgd(data,
                    labels,
                    coefficients,
                    num_epochs=None,
                    batch_size=1,
                    optimizer=None,
                    learning_rate=0.00001,
                    threshold=.00001,
                    max_iter=1000,
                    log_rate=0):
    """Runs sgd where parameters are updated once per epoch.
    The default batch size is 1, which yields standard online
    sgd."""

    assert optimizer is not None

    # Initialize weights
    weights = np.array([np.random.rand() for _ in range(coefficients.size)])
    old_weights = np.array([100 for _ in range(coefficients.size)])

    logging.debug("Initialized weights")

    if labels.size % batch_size != 0:
        batch_size = labels.size / (labels.size // batch_size)

    if num_epochs is None:
        num_epochs = labels.size // batch_size

    weight_hist = []
    idx = 1
    epochs_seen = 0
    num_data_pts_seen = 0
    for epoch_num in range(num_epochs):
        if idx > max_iter or np.allclose(old_weights, weights):
            break

        # get batch
        length = data.shape[0]
        batch_indices = np.random.choice(length,
                                         batch_size)

        mini_batch = np.concatenate((data, labels.reshape(labels.size, 1)),
                                    axis=1)[batch_indices] 
        # run optimization on mini-batch
        old_weights = weights
        weights = optimizer(mini_batch, weights, learning_rate=learning_rate)

        weight_hist.append(weights)
        idx += 1
        if log_rate is not 0 and idx % log_rate == 0:
            print("[{}]: (standard)norm_diff -> {}".format(idx,
                np.linalg.norm(weights - old_weights)))
        epochs_seen += 1
        num_data_pts_seen += length
    return (idx, epochs_seen, num_data_pts_seen, np.array(weights))

def extraction_based_sgd(data,
                         labels,
                         coefficients,
                         batch_size=1,
                         optimizer=None,
                         learning_rate=0.00001,
                         threshold=.00001,
                         max_iter=1000,
                         extraction_threshold=.001,
                         max_extract_iter=10):
    """Runs sgd (`optimizer`) on the same data until the difference in
    `weights` is less than `extraction_threshold`, at which point it moves
    on to the next data.
    """
    weights = np.array([np.random.rand() for _ in range(coefficients.size)])
    old_weights = np.array([100 for _ in range(coefficients.size)])
    idx = 1
    weight_hist = []

    while idx < max_iter:
        batch_indices = np.random.choice(labels.size, batch_size, replace=False)
        if len(batch_indices) < batch_size:
            break

        mini_batch = np.concatenate((data, labels.reshape(labels.size, 1)),
                                    axis=1)[batch_indices]

        extraction_idx = 1
        while extraction_idx < max_extract_iter:
            if np.linalg.norm(old_weights - weights) < extraction_threshold:
                break
            old_weights = weights
            weights = optimizer(mini_batch, weights, learning_rate=learning_rate)
            weight_hist.append(weights)
            extraction_idx += 1
        idx += extraction_idx
    return (idx, np.array(weights), weight_hist)


def sgd(data, weights, learning_rate=0.0001):
    """Returns updated weights based on sgd.

    Parameters:
        data: tuple consisting of (data, label) where `data` is a numpy array
        weights numpy array containing estimated weights

    Return: numpy array containing updated weights
    """
    x_vals = data[:,:-1]

    y = data[:,-1]
    gradient = 0
    for i in range(y.size):
        # Get diff between true y and estimated y
        y_hat = evaluate_function(x_vals[i,:], weights)
        diff = y_hat - y[i]

        # Get gradient
        gradient += get_gradient(diff, x_vals[i,:])
        
    gradient /= y.size

    # Perform SGD update
    weights = weights - learning_rate * gradient

    return weights

def print_result(result_type,
                 n_iter,
                 n_epochs,
                 weights,
                 coefficients,
                 time):
    """Helper function to print results in a formatted manner"""
    output = ("[{res_type}]    n_dim: {dim} max_error: {m_err} runtime: {time} seconds "
            "n_iter: {n_iter} n_epochs: {n_epoch}").format(
                    res_type=result_type,
                    dim=coefficients.size,
                    m_err=np.max(np.abs(weights-coefficients)),
                    time=time,
                    n_iter=n_iter,
                    n_epoch=n_epochs)
    print(output)


def run_test(num_tests=None,
             num_repetition=10,
             size_range=(10,100),
             batch_count=5):
    """Runs a set of tests, stepping up the test size each iteration"""

    if num_tests is None:
        num_tests = size_range[1]//size_range[0]

    test_step = (size_range[1] - size_range[0])//num_tests

    print("Starting tests...")

    for test_num in range(1, num_tests + 1):
        # get test size
        test_size = test_step*test_num
        
        # generate coefficients and data
        coefficients = create_coefficients(test_size)
        n_data = 1000*test_size
        data, labels = generate_labeled_data(n_data, coefficients)

        # set hyperparameters
        n_epochs = n_data
        learning_rate = 0.000001
        
        print("\nRunning tests of parameter size {} and {} data points".format(
            test_size, n_data))

        # Run tests where repeat num goes from 1 to 10
        # and batch size steps through the range 1 to
        # max batch size, steps of factors of 10
        for batch_num in range(1, batch_count + 1):
            batch_size = 10 * batch_num
            print("\nRunning tests for batch size {}".format(batch_size))

            for repeat_num in range(1, 11):
                print("\nRunning tests with RSGD repeat parameter {}".format(repeat_num))

                for mult_test_count in range(num_repetition):
                    with Timer() as repeat_timer:
                        (repeat_iter, repeat_epochs, repeat_pts, repeat_w) = \
                               repeated_epoch_sgd(data,
                                                  labels,
                                                  coefficients,
                                                  num_epochs=n_epochs,
                                                  batch_size=batch_size,
                                                  repeat_num=repeat_num,
                                                  optimizer=sgd,
                                                  learning_rate=learning_rate,
                                                  log_rate=0)
                    print_result("RSGD",
                                 repeat_iter,
                                 repeat_epochs,
                                 repeat_w,
                                 coefficients,
                                 repeat_timer.interval)

                    with Timer() as std_timer:
                        (std_iter, std_epochs, std_pts, std_w) = epoch_based_sgd(data,
                                                                                 labels,
                                                                                 coefficients,
                                                                                 num_epochs=n_epochs,
                                                                                 batch_size=batch_size,
                                                                                 optimizer=sgd,
                                                                                 learning_rate=learning_rate/repeat_num,
                                                                                 log_rate=0)
                    print_result("SGD",
                                 std_iter,
                                 std_epochs,
                                 std_w,
                                 coefficients,
                                 std_timer.interval)

if __name__ == "__main__":
    run_test()

