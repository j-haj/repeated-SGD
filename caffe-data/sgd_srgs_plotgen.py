# Author: Jeff Hajewski
# Date: 4/11/2017

from data_parser import CaffeParser
import matplotlib.pyplot as plt



def plot_lists(x1, l1, x2, l2, label1, label2, title=None, savefile=None):
    plt.clf()
    default_weight = 0.5
    plt.plot(x1, l1, label=label1, lw=default_weight)
    plt.plot(x2, l2, label=label2, lw=default_weight)
    plt.legend()
    if savefile is not None:
        plt.savefig(savefile)
    else:
        plt.show()

def main():
    p = CaffeParser()
    time1, sgd_acc, sgd_loss = p.parse_accuracy_loss("mnist_sgd_mb256_r1.out")
    time2, srgd_acc, srgd_loss = p.parse_accuracy_loss("mnist_sgd_mb256_r2.out")
    plot_lists(time1, sgd_acc, time2, srgd_acc, "SGD", "SRGD ($r=2$)",
            "Loss for MNIST (batch = 256)", "mnist_loss.png")
    plot_lists(time1, sgd_loss, time2, srgd_loss, "SGD", "SRGD ($r=2$)",
            "Accuracy for MNIST (batch = 256)", "mnist_acc.png")
    #cifar_sgd_acc, cifar_sgd_loss = p.parse_accuracy_loss("cifar_quick_sgd.out")
    #cifar_srgd_acc, cifar_srgd_loss = p.parse_accuracy_loss("cifar_quick_srgd.out")
    #plot_lists(cifar_sgd_acc, cifar_srgd_acc, "SGD", "SRGD",
    #        "Loss for CIFAR-10 dataset", "cifar_loss.png")
    #plot_lists(cifar_sgd_loss, cifar_srgd_loss, "SGD", "SRGD",
    #        "Accuracy for CIFAR-10 dataset", "cifar_acc.png")


if __name__ == "__main__":
    main()
