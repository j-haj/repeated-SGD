# Author: Jeff Hajewski
# 4/11/2017

from __future__ import absolute_import, division, print_function
from builtins import (bytes, str, open, super, range, zip,
                      round, input, int, pow, object)

import os
import re

import matplotlib.pyplot as plt
from sklearn.externals.joblib import Memory


class CaffeParser():

    def __init__(self, filepath=None):
        self.filepath = filepath
        pass


    def parse(self, filename):
        
        pass
    
    def parse_accuracy_loss(self, filepath=None):
        """
        Parses the give file (``filename``) for accuracy and loss data

        Args:
            filename: name of the file (with path)

        Return:
            a tuple containing accuracy and loss data where the datapoint index
            corresponds to the iteration number - 1 (e.g., index 0 is for
            iteration 1)
        """
        filename = self.filepath if self.filepath is not None else filepath
        if filename is None:
            # filename cannot be ``None``
            raise TypeError

        accuracies = []
        losses = []
        times = []
        is_first_time = True
        prior_time = 0
        t_index = 0
        with open(filename, 'r') as in_file:
            it_line_pattern = r"solver.cpp:352"
            al_line_pattern = r"solver.cpp:419"
            for line in in_file:
                # Check if the line contains iteration, accuracy, or loss data
                is_iteration_line = re.search(it_line_pattern, line) is not None
                is_al_line = re.search(al_line_pattern, line) is not None
                if not is_iteration_line and not is_al_line:
                    continue
                # Split the line into parts
                split_line = line.strip().split()
                if is_iteration_line:
                    it_index = int(split_line[split_line.index("Iteration") +
                        1].strip(","))

                    # Get time
                    cur_time = split_line[1]
                    if is_first_time:
                        is_first_time = False
                        prior_time = cur_time
                        times.append(0)
                    else:
                        times.append(times[t_index - 1] + time_dif(cur_time, prior_time))
                    prior_time = cur_time
                    t_index += 1
                elif is_al_line:
                    if re.search(r"accuracy", line) is not None:
                        # Accuracy line
                        accuracy = float(split_line[-1])
                        accuracies.append(accuracy)
                    else:
                        # Loss line
                        loss = float(split_line[split_line.index("loss")\
                                + 2])
                        losses.append(loss)
        return (times, accuracies, losses)

    # We want all data lines that have either "START" or "STOP" in them. These
    # lines have keyboards as well, which correspond to the time method.
    def parse_start_stop(self, filepath=None):
        filename = self.filepath if self.filepath is not None else filepath
        if filename is None:
            # ``filename`` cannot be ``None``
            raise TypeError

        lines = []
        with open(filename, 'r') as in_file:
            start_stop_pattern = r"START|STOP"
            for line in in_file:
                # Make sure we only look at lines containing "START" or "STOP"
                if re.search(start_stop_pattern, line) is None:
                    continue

                # Create a tuple containing the timestamp, the label, and
                # whether it's a start or stop line
                split_line = line.strip().split()
                timestamp = split_line[1]
                start_stop = split_line[-1]
                label = split_line[-2]
                lines.append((timestamp, label, start_stop))

        return lines

def plot_list(l):
    x = [i + 1 for i in range(len(loss))]
    plt.plot(x, l)
    plt.show()

def time_dif(t1, t2):
    """
    Returns the difference in t1 and t2 via t1 - t2, in seconds
    """
    split_t1 = t1.split(':')
    split_t2 = t2.split(':')
    diff = (float(split_t1[0]) - float(split_t2[0]))*3600 + \
            (float(split_t1[1]) - float(split_t2[1]))*60 +\
            float(split_t1[2]) - float(split_t2[2])
    return diff

if __name__ == "__main__":
    p = CaffeParser("srgd.out")
    acc, loss = p.parse_accuracy_loss()
    #plot_list(loss)
    plot_list(acc)
