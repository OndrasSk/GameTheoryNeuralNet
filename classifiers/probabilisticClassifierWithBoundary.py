from classifiers.probabilisticClassifier import *
import numpy as np
from abc import ABC, abstractmethod
from scipy.special import expit

class ProbabilisticClassifierWithBoundary(ProbabilisticClassifier):

    def __init__(self, soft=False, b=50, **kwds):
        super().__init__(**kwds)

        self.b = b
        if soft:
            self.classify = self.classify_soft
            self.classify_all = self.classify_all_soft
            self.classify_last = self.classify_last_soft
            self.classify_by_one_classifier = self.classify_by_one_classifier_soft

    def classify_all_soft(self, point):
        """
        Evaluate point on all classifiers using soft logistic classification.

        Parameters
        ----------
        point : vector
            point to be evaluated

        Returns
        -------
        np.vector
            zero-one vector, zero for positive, one for negative
        """
        point = point.reshape(1,-1)
        return np.fromiter(map(lambda x:self.classify_by_one_classifier_soft(point, x), self.classifiers), dtype=np.float64)

    def classify_soft(self, point):
        """
        Probabilistic evaluation on classifiers using soft logistic classification.

        Parameters
        ----------
        point : vector
            point to be evaluated

        Returns
        -------
        float
            probability of negative class
        """
        if self.probabilities.shape[0] == len(self.classifiers):
            return self.classify_all_soft(point).dot(self.probabilities)

    def classify_last_soft(self, point):
        """
        Evaluate point on the last classifier using soft logistic classification.

        Parameters
        ----------
        point : vector
            point to be evaluated

        Returns
        -------
        int
            zero for positive, one for negative
        """
        point = point.reshape(1,-1)
        return self.classify_by_one_classifier_soft(point, self.classifiers[-1])

    def classify_by_one_classifier_soft(self, point, classifier):
        """
        Evaluate point on the classifier using soft logistic classification.

        Parameters
        ----------
        point : np.array(1,n)
            point to be evaluated
        classifier
            classifier on which the point has to be evaluated

        Returns
        -------
        float
            number from <0, 1> interval - zero for positive, one for negative
            It is recommended to use soft classification. Strictly 0-1 classifier can break a DO optimizer
        """
        return expit(-self.b * self.distance_from_boundary(classifier, point))

    @abstractmethod
    def distance_from_boundary(self, classifier, point):
        """
        Returns distance of point from classification boundary

        Parameters
                ----------
        points : np.array(n,d)
                    point to be evaluated
        classifier
            classifier on which the point has to be evaluated

        Returns
        -------
        float
            distance from boundary
        """
        pass