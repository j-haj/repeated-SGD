from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import time
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


# Build network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


# -----------------------------------------------------------------------------
#
# Train/Test functions
#
# -----------------------------------------------------------------------------

def train_model_cifar(epochs, r=1):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=True, download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))
                             ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3801))
            ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    # Create model
    model = Net()
    optimizer = optim.Adam(model.parameters())

    # Put model in training mode
    model.train()
    start = time.time()

    time_steps = []
    loss_vals = []
    aggregate_runtime = 0
    
    for epoch_num in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            for i in range(r):
                start = time.time()

                optimizer.zero_grad()

                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()

                stop = time.time()
                aggregate_runtime += stop - start
                time_steps.append(aggregate_runtime)
                loss_vals.apply(loss.data[0])

    return time_steps, loss_vals

def train_model(epochs, r=1):
    # Load Data
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
   
    # Create model
    model = Net()
    optimizer = optim.Adam(model.parameters())
    # Put model in training mode
    model.train()
    start = time.time()

    time_steps = []
    loss_vals = []
    aggregate_runtime = 0
    for epoch_num in range(1, epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            
            for i in range(r):
                start = time.time()

                # TODO: What does this do?
                optimizer.zero_grad()

                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()

                stop = time.time()
                aggregate_runtime += stop - start
                time_steps.append(aggregate_runtime)
                loss_vals.append(loss.data[0])

    return time_steps, loss_vals

# -----------------------------------------------------------------------------
#
# Graph generation
#
# -----------------------------------------------------------------------------

def generate_graph(x1, y1, label1, x2, y2, label2, title, filename, lw=0.25):
    plt.plot(x1, y1, label=label1, linewidth=lw)
    plt.plot(x2, y2, label=label2, linewidth=lw)
    plt.ylim(0, 2)
    plt.title(title)
    plt.legend()
    plt.savefig(filename)

# -----------------------------------------------------------------------------
#
# Main
#
# -----------------------------------------------------------------------------
def main():

    std_t, std_loss = train_model(args.epochs)
    r2_t, r2_loss = train_model(args.epochs, r=2)
    generate_graph(std_t, std_loss, "SGD", r2_t, r2_loss, "SRGD (r=2)",
    "Loss of SGD vs SGD(r=2)", "test.png")

if __name__ == "__main__":
    # Training settings
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)


    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    main()
