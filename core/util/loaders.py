from enum import Enum
from pathlib import Path
import struct
import numpy as np
import os
import logging

logger = logging.getLogger()


class Data(Enum):
    TRAIN = 0
    TEST = 1

class MNISTLoader():
    """
    Loads the training and testing data for the MNIST dataset
    """
    
    def __init__(self, path="data/mnist"):
        self.path = os.path.normpath(os.getcwd() + path)
        logger.debug("Setting data search path to: {}".format(
            self.path))

    def load_data(self, dtype):
        """Loads data
        
        Parameter:
            dtype: an enum of type Data (Data.TRAIN or Data.Test)

        Return: a tuple of numpy arrays whose first element is an array of
                image data and second element is an array of label data
        """
        
        logger.debug("Loading data from directory: {}".format(self.path))

        # Get file names
        if dtype == Data.TRAIN:
            filename_img = os.path.join(self.path, "train-images-idx3-ubyte")
            filename_label = os.path.join(self.path, "train-labels-idx1-ubyte")
        else:
            filename_img = os.path.join(self.path, "t10k-images-idx3-ubyte")
            filename_label = os.path.join(self.path, "t10k-labels-idx1-ubyte")

        # Load label data
        with open(filename_label, "rb") as label_reader:
            magic, num = struct.unpack(">II", label_reader.read(8))
            label = np.fromfile(label_reader, dtype=np.uint8)
        # Load image data
        with open(filename_img, "rb") as img_reader:
            magic, num, rows, cols = struct.unpack(">IIII", img_reader.read(16))
            img = np.fromfile(img_reader, dtype=np.uint8).reshape(len(label), rows, cols)

        return (img, label)
