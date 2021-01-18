import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from experiments.NN_adversarialAttacksData import getDataDim1NN_10_Simple, getDataDim1NN_10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class Model(nn.Module):
    def __init__(self, dim, h_size):
        super().__init__()
        self.hidden = nn.Linear(dim, h_size)
        self.output = nn.Linear(h_size, 1)

    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        x = self.output(x)
        x = torch.sigmoid(x)
        return x


class Loss_Cross_weight_model(nn.Module):
    def __init__(self):
        super(Loss_Cross_weight_model, self).__init__()

    def forward(self, weight, output, target):
        return F.binary_cross_entropy(output, target, weight=weight, size_average=True)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


model_dnn_2 = nn.Sequential(Flatten(), nn.Linear(784, 200), nn.ReLU(),
                            nn.Linear(200, 10)).to(device)

mnist_train = datasets.MNIST("../data", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST("../data", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=100, shuffle=False)

model_dnn_2.load_state_dict(torch.load("model_dnn_2.pt"))


def trainDNNModels():
    def epoch(loader, model, opt=None):
        total_loss, total_err = 0., 0.
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            yp = model(X)
            loss = nn.CrossEntropyLoss()(yp, y)
            if opt:
                opt.zero_grad()
                loss.backward()
                opt.step()

            total_err += (yp.max(dim=1)[1] != y).sum().item()
            total_loss += loss.item() * X.shape[0]
        return total_err / len(loader.dataset), total_loss / len(loader.dataset)

    opt = optim.SGD(model_dnn_2.parameters(), lr=1e-1)
    for _ in range(10):
        train_err, train_loss = epoch(train_loader, model_dnn_2, opt)
        test_err, test_loss = epoch(test_loader, model_dnn_2)
        print(*("{:.6f}".format(i) for i in (train_err, train_loss, test_err, test_loss)), sep="\t")

    torch.save(model_dnn_2.state_dict(), "model_dnn_2.pt")


def fgsm(model, X, y, epsilon):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    #print(delta)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    #print(loss.item())
    loss.backward()
    #print(delta.grad.detach())
    return epsilon * delta.grad.detach().sign()


def fgsm_framework(model, criterion, weights, X, y, epsilon):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    loss = criterion(weights, model(X + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()


def pgd(model, X, y, epsilon, alpha, numIter):
    """ Construct PGD adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(numIter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + X.shape[0] * alpha * delta.grad.data).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()


def pgd_linf(model, X, y, epsilon, alpha, numIter):
    """ Construct PGD adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(numIter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()


def pgd_linf_framework(model, lossFunc, weights, X, y, epsilon, alpha, numIter):
    """ Construct PGD adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(numIter):
        loss = lossFunc(weights, model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()


def epoch_adversarial(model, loader, attack, *args):
    total_loss, total_err = 0., 0.
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        delta = attack(model, X, y, *args)
        yp = model(X + delta)
        loss = nn.CrossEntropyLoss()(yp, y)

        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
    return total_err / len(loader.dataset), total_loss / len(loader.dataset)


def FGSMAttackTest():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        break

    def plot_images(X, y, yp, M, N):
        f, ax = plt.subplots(M, N, sharex=True, sharey=True, figsize=(N, M * 1.3))
        for i in range(M):
            for j in range(N):
                ax[i][j].imshow(1 - X[i * N + j][0].cpu().numpy(), cmap="gray")
                title = ax[i][j].set_title("Pred: {}".format(yp[i * N + j].max(dim=0)[1]))
                plt.setp(title, color=('g' if yp[i * N + j].max(dim=0)[1] == y[i * N + j] else 'r'))
                ax[i][j].set_axis_off()
        plt.tight_layout()
        plt.show()

    ### Illustrate original predictions
    yp = model_dnn_2(X)
    #plot_images(X, y, yp, 5, 9)

    ### Illustrate attacked images
    delta = fgsm(model_dnn_2, X, y, 0.2)
    yp = model_dnn_2(X + delta)
    #plot_images(X + delta, y, yp, 5, 9)

    print("2-layer DNN:", epoch_adversarial(model_dnn_2, test_loader, fgsm, 0.1)[0])
    print("2-layer DNN:", epoch_adversarial(model_dnn_2, test_loader, fgsm, 0.2)[0])

    ### Illustrate attacked images
    delta = pgd(model_dnn_2, X, y, 0.1, 1e4, 1000)
    yp = model_dnn_2(X + delta)
    #plot_images(X + delta, y, yp, 5, 9)

    print("2-layer DNN:", epoch_adversarial(model_dnn_2, test_loader, pgd_linf, 0.1, 1e-2, 40)[0])
    print("2-layer DNN:", epoch_adversarial(model_dnn_2, test_loader, pgd_linf, 0.2, 1e-2, 40)[0])


def nNBasicFrameworkDim1(hidden_size, dataSource):
    frameworkNN = Model
    loss = Loss_Cross_weight_model
    optimizer = optim.SGD
    lr = 1e-2
    epochs = 1000
    hidden_size = hidden_size

    data, validation_data, labels, weights = dataSource()

    newNeuralNet = frameworkNN(data.size(1), hidden_size)
    optimizer = optimizer(newNeuralNet.parameters(), lr=lr)
    lossFunc = loss()
    newNeuralNet.train()

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        output = newNeuralNet(data)
        loss = lossFunc(weights, output, labels)
        if epoch % 100 == 0:
            print(f"loss = {loss.item()}")
        loss.backward()
        optimizer.step()

    output = newNeuralNet(validation_data)
    loss = lossFunc(weights, output, labels)
    print(f"loss = {loss.item()}")
    print(f"output = {output.data.numpy()}")

    testdata2 = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]])
    testdata2 = Variable(torch.from_numpy(testdata2).float())
    output2 = newNeuralNet(testdata2)
    print(f"output2 = {output2.data.numpy()}")

    X3 = np.array([[10]])
    X3 = Variable(torch.from_numpy(X3).float())
    y3 = np.array([[0]])
    y3 = Variable(torch.from_numpy(y3).float().view(-1, 1))
    weights3 = np.array([[1]])
    weights3 = torch.from_numpy(weights3).float().view(-1, 1)
    output3 = newNeuralNet(X3)
    print(f"output3 = {output3.data.numpy()}")

    delta = fgsm_framework(newNeuralNet, lossFunc, weights3, X3, y3, 6)
    print(X3 + delta)
    output3Adversial = newNeuralNet(X3 + delta)
    print(f"FGSM_Adver = {output3Adversial.data.numpy()}")

    delta = pgd_linf_framework(newNeuralNet, lossFunc, weights3, X3, y3, 10, 1e-2, 1500)
    print(X3 + delta)
    output3Adversial = newNeuralNet(X3 + delta)
    print(f"PGD_Adver = {output3Adversial.data.numpy()}")


if __name__ == "__main__":
    # trainDNNModels()

    #FGSMAttackTest()
    nNBasicFrameworkDim1(10, getDataDim1NN_10_Simple)