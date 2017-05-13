import matplotlib.pyplot as plt

def get_data_from_file(filename):
    time = [[]]*10
    loss = [[]]*10
    with open(filename, 'r') as ifile:
        for line in ifile:
            split_line = line.strip().split(',')
            time[int(split_line[2])].append(float(split_line[0]))
            loss[int(split_line[2])].append(float(split_line[1]))
    return time, loss

def create_avg_line_chart(d1, d2, label1, label2, title, filename):

    x1, y1 = [], []
    x2, y2 = [], []

    t1, l1 = d1
    t2, l2 = d2

    for i in range(len(t1[0])):
        sum_x1, sum_y1, sum_x2, sum_y2 = 0, 0, 0, 0
        for j in range(len(t1)): 
            sum_x1 += t1[j][i]
            sum_y1 += l1[j][i]

            sum_x2 += t2[j][i]
            sum_y2 += l2[j][i]
        x1.append(sum_x1/len(t1))
        y1.append(sum_y1/len(t1))
        x2.append(sum_x2/len(t2))
        y2.append(sum_y2/len(t2))

    plt.plot(x1, y1, label=label1, lw=0.3)
    plt.plot(x2, y2, label=label2, lw=0.3)
    plt.xlim(0, sum(x1)/len(x1))
    plt.title(title)
    plt.legend()
    plt.savefig(filename, dpi=1000)

def create_line_charts(d1, d2, label1, label2, title, filename):
    """
    Read in data from ``files`` and create charts with the given data
    """
   
    # Create the plot object

    # Add first data set to plot
    t1, l1 = d1
    for i in range(len(t1)):
        x, y = t1[i], l1[i]
        plt.plot(x, y, 'b', label=label1, marker='1',ms=0.2,lw=0.35, alpha=0.2)

    # Add second data set to plot
    t2, l2 = d2
    for i in range(len(t2)):
        x, y = t2[i], l2[i]
        plt.plot(x, y, 'g', label=label2, marker='2', ms=0.2, lw=0.35, alpha=0.2)

    plt.title(title)
    plt.legend()
    plt.savefig(filename, dpi=1000)

def plot_run(n, d1, d2, label1, label2, title, filename):
    plt.clf()
    t1, l1 = d1
    t2, l2 = d2

    plt.plot(t1[n], l1[n], label=label1, lw=0.35)
    plt.plot(t2[n], l2[n], label=label2, lw=0.35)

    plt.legend()
    plt.title(title)
    plt.xlim(0, 50)
    plt.savefig(filename)

if __name__ == "__main__":
    files = ["data/pytorch_testMNIST_r10_1494606444.372261.csv",
            "data/pytorch_testMNIST_r1_1494604213.709337.csv",
            "data/pytorch_testMNIST_r2_1494604310.3732932.csv",
            "data/pytorch_testMNIST_r3_1494604394.554535.csv",
            "data/pytorch_testMNIST_r5_1494604393.506811.csv"]

    r1_data = get_data_from_file(files[1])
    r2_data = get_data_from_file(files[2])
    r3_data = get_data_from_file(files[3])
    r5_data = get_data_from_file(files[4])
    r10_data = get_data_from_file(files[0])
    create_avg_line_chart(r1_data, r2_data, "r = 1", "r = 2",
            "Comparison of r = 1 vs r = 2", "r1_r2_comparison.png")

    plot_run(0, r1_data, r2_data, "r = 1", "r = 2",
            "Comparison of r = 1 vs r = 2", "r1_r2_single_comp.png")

