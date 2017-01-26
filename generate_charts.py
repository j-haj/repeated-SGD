import logging
import matplotlib.pyplot as plt

logger = logging.getLogger()

def load_data(filename):
    """Loads data from filename, which is a space separated file

    File format:

    <optimizer> <dimension> <r> <batch_size> <num_iteration> <runtime> <e1> <e2>
    
    Return: loaded data as dictionary whose keys represent the optimizer used
    """
    # Create data dict template
    data = {"sgd": {}, 
        "srgd": {},
        "assgd": {},
        "assrgd": {}}
    with open(filename, 'r') as reader:
        for row in reader:
            vals = row.strip().split()
            key = vals[0]
            dim = int(vals[1])
            r = int(vals[2])
            batch_sz = int(vals[3])
            n_iter = int(vals[4])
            time = float(vals[5])
            err1 = float(vals[6])
            err2 = float(vals[7])
            if r not in data[key]:
                data[key][r] = {
                        "batch_size": [],
                        "n_iter": [],
                        "time": [],
                        "norm_err": [],
                        "max_err": []}

            data[key][r]["batch_size"].append(batch_sz)
            data[key][r]["n_iter"].append(n_iter)
            data[key][r]["time"].append(time)
            data[key][r]["norm_err"].append(err1)
            data[key][r]["max_err"].append(err2)
    return data

def generate_scatter_err_vs_time(results, plt_name):
    """Generates two scatter plots showing error
    as a function of runtime for the various methods.
    """
    fig, ax = plt.subplots()
    
    # Set labels and title
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Norm Error")
    ax.set_title("Error vs. Time")

    # Get x,y pairs
    plt_data = {"sgd": {},
            "srgd": {},
            "assrgd": {},
            "assgd": {}}

    r_vals = [1, 5, 10]
    
    # Create marker and color dictionary
    styles = {"sgd": {"marker": ".", "color": "red"},
            "srgd - 1": {"marker": "+", "color": "blue"},
            "srgd - 5": {"marker": "+", "color": "green"},
            "srgd - 10": {"marker": "+", "color": "orange"},
            "assrgd - 1": {"marker": "x", "color": "blue"},
            "assrgd - 5": {"marker": "x", "color": "green"},
            "assrgd - 10": {"marker": "x", "color": "orange"},
            "assgd": {"marker": 10, "color": "black"}
            }

    for r in r_vals:
        for key in plt_data:
            plt_data[key][r] = {"x": 0, "y": {"norm": 0, "max": 0}}
            # Get x vals
            plt_data[key][r]["x"] = results[key][r]["time"] 
            #sum(results[key][r]["time"])/len(results[key][r]["time"])

            # Get y vals
            plt_data[key][r]["y"]["norm"] = results[key][r]["norm_err"]
            #sum(results[key][r]["norm_err"])/len(results[key][r]["norm_err"])

            plt_data[key][r]["y"]["max"] = \
            sum(results[key][r]["max_err"])/len(results[key][r]["max_err"])

    for key in plt_data:
        if "r" in key:
            # One of the repeat optimizers
            for r in r_vals:
                label = key + " - " + str(r)
                ax.scatter(plt_data[key][r]["x"],
                        plt_data[key][r]["y"]["norm"],
                        label=label,
                        marker=styles[label]["marker"],
                        color=styles[label]["color"])
        else:
            # Non-repeat optimizes
            ax.scatter(plt_data[key][1]["x"],
                    plt_data[key][1]["y"]["norm"],
                    label=key,
                    marker=styles[key]["marker"],
                    color=styles[key]["color"])
    ax.legend()
    ax.grid(True)

    plt.savefig(plt_name)


def generate_chart():
    print("Called this function successfully")
    logger.info("Generating chart")
    logger.error("Generating chart - error!!")

if __name__ == "__main__":
    data = load_data("results_10_0001.csv")
    generate_scatter_err_vs_time(data, "dim-10.png")
