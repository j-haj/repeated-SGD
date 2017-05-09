import mnist_loader
import network

def main():
    """MAIN"""
    training_data, validation_data, test_data = \
            mnist_loader.load_data_wrapper()
    net = network.Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 1.0, test_data=test_data)

if __name__ == "__main__":
    main()
