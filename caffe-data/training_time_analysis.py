# Author: Jeff Hajewski
#
# Analyze the time spent in various stages of training in a caffe model

import matplotlib.pyplot as plt
import os

from collections import deque
from data_parser import CaffeParser
from sklearn.externals.joblib import Memory

mem = Memory("__datacache__")

def compute_time_diff(time1, time2):
    """
    Takes two times (as strings) and computes their difference, returns a
    float that is the difference in seconds . Time format is assumed to be
    of the form:
        HH:MM:SS.SUBSEC
    Note that the difference is calculated ``time2`` - ``time1``

    Args:
        time1: a time of the described format
        time2: a time of the described format

    Returns:
        Time difference in seconds
    """
    t1_comps = time1.split(':')
    t2_comps = time2.split(':')
    h_dif = []
    for (a, b) in zip(t1_comps, t2_comps):
        h_dif.append(float(b) - float(a))
    return h_dif[0]*60**2 + h_dif[1]*60 + h_dif[2]



def analyze_runtime_distribution(data):
    """
    Analyzes ``data``, extracting unique labels from ``data`` and determining
    what percentage of the total runtime each label makes up. Note that ``data``
    is assumed to be a list of tuples, where each tuple has the format:
        (<timestamp>, <label>, <START/STOP>)
    and timestamps have the format:
        timestamp := HH:MM:SS.SUBSEC

    Args:
        data: a list of tuples, where each tuple has the format (<timestamp>,
        <label>, <START/STOP>)

    Returns:
        A dictionary whose keys are labels and whose values are total runtimes
    """
    
    # First let's get unique labels
    unique_labels = {x[1] for x in data}

    # Now create a dictionary whose keys are unique_labels and whose values are
    # queues of timestamps
    queue_collection = dict.fromkeys(unique_labels, deque())
    queue_times = dict.fromkeys(unique_labels, 0)
    for (timestamp, label, flag) in data:
        if flag == "START":
            queue_collection[label].append(timestamp)
        else:
            prior_timestamp = queue_collection[label].popleft()

            # Parse timestamps
            time_dif = compute_time_diff(prior_timestamp, timestamp)
            queue_times[label] += time_dif
    print(queue_times)
    return queue_times

def generate_pie_chart(data, title, savefile=None):
    """
    Creates a pie chart us the keys and values from the input dictionary
    ``data``

    Args:
        data: a dictionary whose keys are label strings and whose values are
            representing total runtime
    """
    labels, vals = [], []
    for (key, value) in data.items():
        labels.append(key)
        vals.append(value)
    plt.pie(vals, labels=labels, autopct="%1.1f%%")
    plt.axis("equal")
    plt.title(title)
    if savefile is not None:
        plt.savefig(savefile)
    else:
        plt.showfigure()

def generate_stacked_bar(data1, data2, label1, label2, title, savefile=None):
    plt.clf()
    fig, ax = plt.figure(), plt.subplot(111)
    to_plot = []
    X = [0, 1]
    for label in data1.keys():
        to_plot.append((label, data1[label], data2[label]))

    print(to_plot)
    prior_val = [0, 0]
    sorted_to_plot = sorted(to_plot, key=lambda x: x[2], reverse=True)
    for datum in sorted_to_plot:
        print("Plotting: {}".format(datum))
        ax.bar(X, [datum[1], datum[2]], label=datum[0], bottom=prior_val,
                align="center")
        prior_val = [datum[1], datum[2]]
    ax.set_title(title)
    lgd = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.5),
            fancybox=True, ncol=5)
    ax.set_xticklabels((label1, label2))
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Runtime (seconds)")
    if savefile is not None:
        plt.savefig(savefile, bbox_extra_artist=(lgd,), bbox_inches="tight")
    else:
        plt.show()

    

def main():
    # load parses data
    p = CaffeParser()

    sgd_input = p.parse_start_stop("mnist_sgd_r2.out")
    srgd_input = p.parse_start_stop("mnist_srgd_r2.out")

    sgd_data = analyze_runtime_distribution(sgd_input)
    srgd_data = analyze_runtime_distribution(srgd_input)

    generate_pie_chart(sgd_data, "SGD Runtime Analysis", "sgd_analysis.png")
    generate_pie_chart(srgd_data, "SRGD Runtime Analysis", "srgd_analysis.png")
    generate_stacked_bar(sgd_data, srgd_data, "SGD", "SRGD", "SGD vs SRGD Runtime",
            savefile="sgd_vs_srgd.png")
if __name__ == "__main__":
    main()
