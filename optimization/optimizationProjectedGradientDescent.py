from abc import abstractmethod

import scipy.optimize as opt
import  numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from optimization.optimizationInterface import OptimizationInterface


class OptimizationProjectedGradientDescent(OptimizationInterface):

    def __init__(self, p_classifier, negative_points, n_points, logger):
        super().__init__(p_classifier, negative_points, n_points, logger)

    def optimize(self, fun, bounds, positive_data=None):
        # choose defenders classifiers to attack
        point = np.array([10])
        arr = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])

        print(f"number of classifiers= {len(self.p_classifier.classifiers)}")
        print(f"index of max= {np.argmax(self.p_classifier.probabilities)}")
        bestClassifier = self.p_classifier.classifiers[np.argmax(self.p_classifier.probabilities)]
        for prob in self.p_classifier.probabilities:
            print(prob)

        print(f"Taking most probable classifier, and using it, its probability: ")
        for p in arr:
            print(f"point = {p}, complete value of BR = {fun(p)}")
            print(f"point = {p}, classify by all classifiers probability = {self.p_classifier.classify(p)}")
            print(f"point = {p}, classify by one classifier (best) = {self.p_classifier.classify_by_one_classifier(p, bestClassifier, False)}")



        # find points to attack
        lastAttackerPoint = positive_data[0][-1]
        print(f"lastAttackerPoint = {lastAttackerPoint}")

        # attack defenders best classifier




        return point, fun(point)

    def pgd_linf(self, model, lossFunc, weights, X, y, epsilon, alpha, numIter):
        """ Construct PGD adversarial examples on the examples X"""
        delta = torch.zeros_like(X, requires_grad=True)
        for t in range(numIter):
            loss = lossFunc(weights, model(X + delta), y)
            loss.backward()
            delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
            delta.grad.zero_()
        return delta.detach()
