import numpy as np
from abc import ABC, abstractmethod

from constants import THETA, UTILITY_STEP
from helpers import is_higher_then_all_in_by


class ProbabilisticClassifier(ABC):
    def __init__(self, hard_fp=1):
        super().__init__()
        self.classifiers = []
        self.probabilities = np.array([])
        self.fp = []
        self.hard_fp = hard_fp
        self.logger = None

    def initialize(self, negative_data, negative_data_utility, positive_data, positive_data_utility, logger,
                   val_data=None):
        """
        Creates initial classifier

        Parameters
        ----------
        negative_data : np.array
            Data to be classified (all labeled 1)
        negative_data_utility : np.array
            utility values of negative data
        positive_data : np.array
            Data to be classified (all labeled 0)
        positive_data_utility : np.array
            utility values of positive data

        validation_data : np.array
            Data to be used for cross-validation
        """
        self.negative_data = negative_data
        self.negative_data_weights = negative_data_utility
        self.val_data = val_data

        self.negative_data = np.vstack(
            (self.negative_data, np.zeros(self.negative_data[0].shape), 10 * np.ones(self.negative_data[0].shape)))
        self.negative_data_weights = np.hstack((self.negative_data_weights, 100, 100))
        self.update(positive_data, positive_data_utility, np.array([1]), np.empty([1, 0]))
        self.negative_data = self.negative_data[:-2, :]
        self.negative_data_weights = self.negative_data_weights[:-2]
        self.logger = logger

    def classify_all(self, point):
        """
        Evaluate point on all classifiers.

        Parameters
        ----------
        point : vector
            point to be evaluated

        Returns
        -------
        np.vector
            zero-one vector, each position in vector is one classifier evaluation of a point,
            zero for positive, one for negative
        """
        point = point.reshape(1, -1)
        return np.fromiter(map(lambda x: self.classify_by_one_classifier(point, x), self.classifiers), dtype=np.float64)

    def classify(self, point):
        """
        Probabilistic evaluation on classifiers.

        Parameters
        ----------
        point : vector
            point to be evaluated

        Returns
        -------
        float
            probability of negative class (negative = benign point)
        """
        if self.probabilities.shape[0] == len(self.classifiers):
            #print(f"self.classify_all(point) = {self.classify_all(point)}")
            #print(
            #    f"self.classify_all(point).dot(self.probabilities) = {self.classify_all(point).dot(self.probabilities)}")
            return self.classify_all(point).dot(self.probabilities)

    def update_support(self, support):
        """
        Sets new probabilities on classifiers.

        Parameters
        ----------
        support : vector
            probabilities of classifiers
        """
        self.probabilities = support

    def support_size(self):
        """
        Gets the size of actual support.

        Returns
        ----------
        int
            number of the classifiers with probability > 0
        """
        return np.sum(self.probabilities > THETA)

    def classify_last(self, point):
        """
        Evaluate point on the last classifier.

        Parameters
        ----------
        point : vector
            point to be evaluated

        Returns
        -------
        int
            zero for positive, one for negative
        """
        point = point.reshape(1, -1)
        return self.classify_by_one_classifier(point, self.classifiers[-1])

    @abstractmethod
    def classify_by_one_classifier(self, point, classifier):
        """
        Evaluate point on the classifier.

        Parameters
        ----------
        points : np.array(n,d)
            point to be evaluated
        classifier
            classifier on which the point has to be evaluated

        Returns
        -------
        float
            number from <0, 1> interval - zero for positive, one for negative
        """
        pass

    def all_classifiers_utilities(self, att_support, game_matrix):
        """
        Gets expected utilities of all classifiers.

        Parameters
        ----------
        att_support : np.array
            Actual attackers strategy
        game_matrix : np.array
            Game matrix

        Returns
        -------
        np.array
            expected utilities of classifiers
        """
        if att_support.shape[0] < game_matrix.shape[0]:
            att_support = np.pad(att_support, (0, 1), 'constant')

        utilities = att_support.dot(game_matrix)

        return utilities

    def get_matrix_column(self, new_classifier, positive_data, positive_data_utility):
        """
        Gets column of game matrix corresponding to the new classifier

        Parameters
        ----------
        new_classifier : classifier
            new classifier for evaluation
        positive_data : np.array
            List of all attackers points
        positive_data_utility : np.array
            Utilities of all attackers points

        Returns
        -------
        np.array
            new column of matrix
        """
        column = np.float64(
            np.apply_along_axis(lambda x: self.classify_by_one_classifier(x.reshape(1, -1), new_classifier),
                                1, positive_data))
        column *= np.float64(positive_data_utility).reshape(-1, 1)
        return column

    def classifier_utility(self, att_support, column):
        """
        Gets expected utility of new classifier.

        Parameters
        ----------
        att_support : np.array
            Actual attackers strategy
        column : np.array
            new column of the matrix corresponding to the newly added classifier

        Returns
        -------
        float
            expected utility of classifier
        """
        new_utility = att_support.dot(column)
        return new_utility

    def expand_classifiers(self, new_classifier, positive_data, positive_data_utility, att_support, game_matrix,
                           utilities=None, new_utility=None, column=None, thld=UTILITY_STEP):

        """
        Expands set of classifiers by a new one, if it has higher utility than thld.

        Parameters
        ----------
        new_classifier : classifier
            new classifier for evaluation
        positive_data : np.array
            List of all attackers points
        positive_data_utility : np.array
            Utilities of all attackers points
        att_support : np.array
            Actual attackers strategy
        game_matrix : np.array
            Game matrix
        utilities : np.array
            expected utilities of classifiers
        new_utility : float
            expected utility of new classifier
        column : np.array
            new column of the matrix corresponding to the newly added classifier
        thld : float
            threshold on difference between utilities

        Returns
        -------
        np.array
            new column of matrix or None, if new classifier is not added
        """

        if utilities is None:
            utilities = self.all_classifiers_utilities(att_support, game_matrix)

        if new_utility is None:
            column = self.get_matrix_column(new_classifier, positive_data, positive_data_utility)
            new_utility = self.classifier_utility(att_support, column)

        if not self.logger is None:
            self.logger.info(f"classifier utilities = {utilities}")
            self.logger.info(f"new classifier utility = {new_utility}")

        try:
            if not self.logger is None:
                self.logger.debug(f"tree {new_classifier.tree_.threshold}")
        except:
            pass

        if is_higher_then_all_in_by(utilities, new_utility,
                                    thld):  # np.all(np.abs(utilities - new_utility) > thld): #distance of utilities
            if not self.logger is None:
                self.logger.info(f"classifier added")
            self.classifiers.append(new_classifier)
            return column
        if not self.logger is None:
            self.logger.info(f"no new classifier added")
        return None

    @abstractmethod
    def update(self, positive_data, positive_data_utility, att_support, game_matrix):
        """
        Finds best classifier for data on the actual support.

        Parameters
        ----------
        positive_data : np.array
            data of positive class
        positive_data_utility : np.array
            utilities of points of positive class
        att_support
            probabilities of positive data - weights of points
        game_matrix
            utility matrix of the game

        Returns
        -------
        np.array
            utilities of points classified by new classifier, None if classifier not added
        """
        pass

    def prepare_data(self, positive_data, positive_data_utility, att_support):
        """
        Prepare data, labels and data weights for the classification

        Parameters
        ----------
        positive_data : np.array
            data of positive class
        positive_data_utility : np.array
            utilities of points of positive class
        att_support
            probabilities of positive data - weights of points

        Returns
        -------
        np.array
            all data for training
        np.array
            all data for validation
        np.array
            labels of data
        np.array
            weights of data
        """
        data = np.concatenate((self.negative_data, positive_data), axis=0)
        if self.val_data is None:
            validation_data = data
        else:
            data = np.concatenate((self.val_data, positive_data), axis=0)
        weights = self.negative_data_weights
        # weights = np.ones(self.negative_data.shape[0])

        att_weights = att_support * positive_data_utility
        weights = np.concatenate((weights, att_weights), axis=0)

        if self.logger:
            self.logger.debug(f"prepare data positive_data_utility = {positive_data_utility}")
            self.logger.debug(f"prepare data att_support = {att_support}")
            self.logger.debug(f"prepare data att_weights = {att_weights}")
            self.logger.debug(f"prepare data weights = {weights}")

        labels = np.ones(self.negative_data.shape[0], dtype=np.int)
        labels = np.concatenate((labels, np.zeros(positive_data.shape[0], dtype=np.int)), axis=0)

        return data, validation_data, labels, weights

    @abstractmethod
    def equals(self, cls1, cls2):
        """
        Compares, if two classifiers equals

        Parameters
        ----------
        cls1 : classifier
            first classifier
        cls 2 : classifier
            second classifier

        Returns
        -------
        bool
            equality of classifiers
        """
        return cls1 == cls2
        # TODO Needs overriding

    def used(self, cls):
        """
        Check, if the same classifier already is in the list

        Parameters
        ----------
        cls : classifier
            first classifier

        Returns
        -------
        bool
            lis contain cls
        """
        return any(map(lambda x: self.equals(cls, x), self.classifiers))

    def __len__(self):
        return len(self.classifiers)

    def get_fp(self):
        """
        Returns the false-positive rate for all classifiers

        Returns
        -------
        float list
            list of all false-positive rates for all classifiers
        """
        if len(self.fp) < len(self.classifiers):
            for clf in self.classifiers[len(self.fp):]:
                classification = self.classify_by_one_classifier(self.negative_data, clf)
                self.fp.append(np.sum(np.abs(1 - classification)) / self.negative_data.shape[0])
                if not self.val_data is None:
                    classification = self.classify_by_one_classifier(self.val_data, clf)
                    self.fp[-1] += np.sum(np.abs(1 - classification)) / self.val_data.shape[0]
        return self.fp
