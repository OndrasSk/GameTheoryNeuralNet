import numpy as np
import torch
from torch.autograd import Variable


def transformForPyTorch(data, validation_data, labels, weights):
    data = Variable(torch.from_numpy(data).float())
    validation_data = Variable(torch.from_numpy(validation_data).float())
    labels = Variable(torch.from_numpy(labels).float().view(-1, 1))
    weights = torch.from_numpy(weights).float().view(-1, 1)

    return data, validation_data, labels, weights


def getDataDim1NN_10_Simple():
    data = np.array(
        [[1.5512166], [0.89052951], [1.3123235], [1.31590415], [1.29282538], [0.5807871], [0.40827217], [0.2922468],
         [0.24817726],
         [1.16178609], [10.]])
    validation_data = np.array(
        [[1.5512166], [0.89052951], [1.3123235], [1.31590415], [1.29282538], [0.5807871], [0.40827217], [0.2922468],
         [0.24817726], [1.16178609], [10.]])
    labels = np.array([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [0]])

    weights = np.array([[1], [1], [1], [1], [1], [1], [1], [1], [1], [1], [10]])
    return transformForPyTorch(data, validation_data, labels, weights)


def getDataDim1NN_10():
    data = np.array(
        [[1.5512166], [0.89052951], [1.3123235], [1.31590415], [1.29282538], [0.5807871], [0.40827217],
         [0.2922468], [0.24817726], [1.16178609],
         [10.], [4.72251944], [2.80292246], [2.09224108], [1.78155724], [1.64032436],
         [1.57972041], [1.56499465], [1.56106092], [1.55889364], [1.55627871], [1.55480682],
         [1.55248746], [1.55024187], [1.54867691], [1.54659952], [1.5455253], [1.54031229],
         [1.53709614], [1.53608392], [1.53467074], [1.53308264], [1.53123891], [1.52748642],
         [1.5242381], [1.52243528], [1.5215245], [1.52128586], [1.52090134], [1.51904939],
         [1.51874179], [1.51832212], [1.51812933], [1.51504155], [1.51480007], [1.5108112],
         [1.51079663], [1.51011091], [1.50783917], [1.50718385], [1.50342917]])
    validation_data = data

    labels = np.zeros((data.shape[0], 1))
    labels[:10, 0] = 1
    print(labels)
    weights = np.ones((data.shape[0], 1))

    return transformForPyTorch(data, validation_data, labels, weights)
