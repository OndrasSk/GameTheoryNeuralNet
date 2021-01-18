import copy

import torch
from torch import nn
from torch.autograd import Variable

from classifiers.probabilisticClassifier import *
from classifiers.notAllowHardFP import *
import torch.nn.functional as F
import torch.optim as optim

from constants import UTILITY_STEP


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


class ProbabilisticNN(ProbabilisticClassifier, NotAllowHardFP):

    def __init__(self, model=Model, hidden_size=10, loss=Loss_Cross_weight_model, optimizer=optim.SGD, lr=1e-2,
                 epochs=1500, iterations=10, soft_classification=False, equality_thresh=1e-8,
                 thld=UTILITY_STEP, gradient=-np.inf, continue_training=False, **kwds):
        super().__init__(**kwds)
        self.model = model
        self.hidden_size = hidden_size
        self.continue_training = continue_training
        self.last_clf = None
        self.loss = loss
        self.optimizer = optimizer
        self.lr = lr
        self.epochs = epochs
        self.iterations = iterations
        self.soft_classification = soft_classification
        self.thld = thld
        self.gradient = gradient
        if equality_thresh != np.inf:
            self.compare = lambda x, y: torch.allclose(x, y, rtol=0, atol=equality_thresh)
        else:
            self.compare = torch.equal

    def classify_by_one_classifier(self, point, classifier, rounding=True):
        point = Variable(torch.from_numpy(point).float())
        res = torch.t(classifier(point)).data.numpy().reshape(-1)
        if not self.soft_classification and rounding:
            res = np.round(res)
        return res

    def equals(self, c1, c2):
        for key_item_1, key_item_2 in zip(c1.state_dict().items(), c2.state_dict().items()):
            if self.compare(key_item_1[1], key_item_2[1]):
                pass
            else:
                return False
        return True

    def update(self, positive_data, positive_data_utility, att_support, game_matrix):
        data, validation_data, labels, weights = self.prepare_data(positive_data, positive_data_utility, att_support)

        att_support1 = att_support * positive_data_utility
        att_support1 = att_support1[weights[(1 - labels).astype(bool)] > THETA]

        validation_data = validation_data[weights > THETA]
        data = data[weights > THETA]
        labels = labels[weights > THETA]
        weights = weights[weights > THETA]

        positive_data_bool = labels == 0
        support_full = np.zeros(positive_data_bool.shape)
        support_full[positive_data_bool] = att_support1

        if self.logger is not None:
            self.logger.debug(f"classifier training data = {data}")
            self.logger.debug(f"classifier validation data = {validation_data}")
            self.logger.debug(f"classifier labels = {labels}")
            self.logger.debug(f"classifier weights = {weights}")

        data = Variable(torch.from_numpy(data).float())
        validation_data = Variable(torch.from_numpy(validation_data).float())
        labels = Variable(torch.from_numpy(labels).float().view(-1, 1))
        weights = torch.from_numpy(weights).float().view(-1, 1)

        if not self.continue_training or self.last_clf is None:
            new_classifier = self.model(data.size(1), self.hidden_size)
            if len(self.classifiers) > 0:
                self.last_clf = new_classifier
        else:
            new_classifier = copy.deepcopy(self.last_clf)

        optimizer = self.optimizer(new_classifier.parameters(), lr=self.lr)
        criterion = self.loss()
        new_classifier.train()
        utilities = self.all_classifiers_utilities(att_support, game_matrix)

        if np.any(utilities <= self.thld):
            return None

        old_loss = 1000
        grads = []
        it = 0
        better = False
        grad = np.inf

        while (not better or grad > self.gradient) and it < self.iterations:
            for epoch in range(1, self.epochs + 1):
                optimizer.zero_grad()
                output = new_classifier(data)
                loss = criterion(weights, output, labels)
                loss.backward()
                optimizer.step()

            output = new_classifier(validation_data)
            loss = criterion(weights, output, labels)
            loss = loss.item()
            output = output.data.numpy()

            new_utility = support_full.dot(np.round(output))
            better = is_higher_then_all_in_by(utilities, new_utility, self.thld)
            if self.logger is not None:
                self.logger.debug(f"classifier utilities = {np.min(utilities)}")
                self.logger.debug(f"new classifier utility = {new_utility}")
                self.logger.debug(f"new classifier is better = {better}")

            if self.gradient != -np.inf:
                grad = old_loss - loss
                grads.append(grad)
                old_loss = loss
            it += 1

        expand = self.expand_classifiers(new_classifier, positive_data, positive_data_utility, att_support,
                                         game_matrix, utilities, thld=self.thld)
        new_classifier.eval()

        return expand
