# first line: 38
@mem.cache
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
