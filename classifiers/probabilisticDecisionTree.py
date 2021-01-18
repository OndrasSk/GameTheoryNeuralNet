from sklearn.tree import DecisionTreeClassifier

from classifiers.notAllowHardFP import NotAllowHardFP
from classifiers.probabilisticClassifierWithBoundary import ProbabilisticClassifierWithBoundary, UTILITY_STEP
import numpy as np

from constants import THETA
from helpers import is_higher_then_all_in_by


class ProbabilisticDecisionTree(ProbabilisticClassifierWithBoundary, NotAllowHardFP):

    def __init__(self, max_depth=None, gradient=None, **kwds):
        super().__init__(**kwds)
        self.max_depth = max_depth
        self.gradient = gradient

    def classify_by_one_classifier(self, point, classifier):
        return classifier.predict(point)

    def equals(self, clf1, clf2):
        return clf1.tree_.node_count== clf2.tree_.node_count and np.all(clf1.tree_.threshold==clf2.tree_.threshold)\
               and np.all(clf1.tree_.feature==clf2.tree_.feature)

    def get_diff(self, output1, output2, labels, weights):
        """
        Get weighted quality difference between classifiers.

        Parameters
        ----------
        output1 : np.array
            points classification given by first classifier
        output2 : np.array
            points classification given by second classifier
        labels : np.array
            correct classification of points
        weights : np.array
            weights of points

        Returns
        -------
        float
            quality difference between classifications
        """
        miss1 = np.abs(labels - output1)
        miss2 = np.abs(labels - output2)
        return (miss1 - miss2).dot(weights)

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

        if self.gradient is None:
            new_classifier = DecisionTreeClassifier(max_depth=self.max_depth)
            new_classifier.fit(data, labels, weights)
        else:
            utilities = self.all_classifiers_utilities(att_support, game_matrix)
            if (np.any(utilities <= UTILITY_STEP)):
                return None

            diff = np.inf
            depth = 1
            new_clf1 = DecisionTreeClassifier(max_depth=depth)
            new_clf1.fit(data, labels, weights)
            output1 = new_clf1.predict(validation_data)

            new_utility = support_full.dot(output1)
            better = is_higher_then_all_in_by(utilities, new_utility, UTILITY_STEP)
            if not self.logger is None:
                self.logger.debug(f"classifier utilities = {np.min(utilities)}")
                self.logger.debug(f"new classifier utility = {new_utility}")
                self.logger.debug(f"new classifier is better = {better}")

            while (diff > self.gradient or not better) and (diff>0):
                depth += 1
                new_clf2 = DecisionTreeClassifier(max_depth=depth)
                new_clf2.fit(data, labels, weights)
                output2 = new_clf2.predict(validation_data)
                diff = self.get_diff(output1, output2, labels, weights)
                output1 = output2
                new_clf1 = new_clf2

                new_utility = support_full.dot(output1)
                better = is_higher_then_all_in_by(utilities, new_utility, UTILITY_STEP)
                if not self.logger is None:
                    self.logger.debug(f"classifier utilities = {np.min(utilities)}")
                    self.logger.debug(f"new classifier utility = {new_utility}")
                    self.logger.debug(f"new classifier is better = {better}")

            new_classifier = new_clf1

        return self.expand_classifiers(new_classifier, positive_data, positive_data_utility, att_support, game_matrix)

    def distance_from_boundary(self, classifier, point):
        ok = classifier.tree_.feature >= 0
        nodes = np.sum(ok)
        return np.min(np.abs(classifier.tree_.threshold[ok] -
                             np.tile(point, (nodes,1))[np.arange(nodes),classifier.tree_.feature[ok]]))