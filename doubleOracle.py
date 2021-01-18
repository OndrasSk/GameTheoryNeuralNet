from GT_solvers import *
from optimization.optimizationDiscretization import OptimizationDiscretization
from helpers import *
from constants import *
from pathlib import Path

import warnings
from classifiers.notAllowHardFP import *

import logging
import os
import sys

from optimization.optimizationBasinhoppingDiscretization import *

import matplotlib.pyplot as plt

#from experiments.time_measure import timer
from optimization.optimizationProjectedGradientDescent import OptimizationProjectedGradientDescent
from timer import Timer

PLOT_ALL = False

class DoubleOracle:
    def __init__(self, negative_data, utility, probabilistic_classifier, bounds, fp_thresh=None, optimizer=None, nfg_solver="cvxopt",  #"gurobi",#
                 step_type="atonce", mixing_weights=np.array([0.8, 0.05, 0.05, 0.05, 0.05]),
                 negative_data_weights_generation="ones", negative_data_weights=None, name="do_output", debug=True):
        """
        Initialize Double Oracle algorithm

        Parameters
        ----------
        negative_data : np.array
            data of negative class (n, d)
            (benign points)
        utility : function(np.array(d)) -> float
            utility function callable on point
        probabilistic_classifier : ProbabilisticClassifier
            class used for classification
        bounds : ((lower_bound_1, upper_dound_1), ..., (lower_bound_d, upper_dound_d))
            bounds of game space
        fp_thresh : float
            threshold for the false-positive constraint
        optimize : OptimizationInterface
            optimization function used by DO
        nfg_solver : ["gurobi", "cvxopt]"
            lp solver to be used by DO
        step_type : ["atonce", "alternating", "mixing"]"
            type of DO iterations: atonce -> both players computes BR at once
                                    alternating -> both players alternates in computation of BR
                                    mixing -> both players computes BR at once on strategy weighted from previous ones
                                             mixing weights are given by mixing_weights
        mixing_weights : np.array(1,history_length)
            array of weights for mixed DO iteration type
        negative_data_weights_generation : ['other', 'uniform', 'ones']
            type of negative weights: other -> uses weights from parameter negative_data_weights
                                      uniform -> uses weight 1/n, where n is number of points
                                      ones -> uses weight 1
        negative_data_weights : np.array(n, d)
            array of weights for negative data
            used if negative_data_weights_generation is "other"
        name : string
            name of directory for outputs
        """
        levels_up = 1
        self.path = str(Path().absolute().parents[levels_up - 1])
        self.init_logger(name, debug)
        #self.timer = timer()
        self.timer = Timer()

        # benign points
        self.negative_data = negative_data
        print("negative data=", negative_data)
        self.utility = utility

        self.p_classifier = probabilistic_classifier

        self.bounds = bounds

        if fp_thresh is None and isinstance(probabilistic_classifier, NotAllowHardFP):
            warnings.warn('With this classifier is not possible use hard FP constrain. Soft FP constraint 0.01 forced.', Warning)
            fp_thresh = 0.01

        self.fp_thresh = fp_thresh

        optim2 = OptimizationProjectedGradientDescent(self.p_classifier, self.negative_data, 10, self.logger)
        self.optimize2 = optim2.optimize

        if optimizer is None:
            optim = OptimizationDiscretization(self.p_classifier, self.negative_data, 10, self.logger)
            self.optimize = optim.optimize
        else:
            optim = optimizer(self.p_classifier, self.negative_data, 10, self.logger)
            self.optimize = optim.optimize

        if nfg_solver=="cvxopt":
            self.solve_nfg = solve_NFG_cvx
        else:
            if nfg_solver!="gurobi":
                warnings.warn(
                    'Invalid LP solver set, algorithm is using default gurobi',
                    Warning)
                self.solve_nfg = solve_NFG_gurobi

        if step_type == "alternating":
            self.mixing_weights = np.array([])
            self.step = self.alternating_step
        elif step_type == "mixing":
            self.step = self.mixing_step
            self.mixing_weights = mixing_weights
        else:
            if step_type != "atonce":
                warnings.warn('Invalid step type, algorithm is using default gurobi', Warning)
            self.mixing_weights = np.array([])
            self.step = self.mixing_step

        if negative_data_weights_generation == "others" and not negative_data_weights is None:
            self.negative_data_weights = negative_data_weights
        elif negative_data_weights_generation == "uniform":
            self.negative_data_weights = np.ones(negative_data.shape[0])/negative_data.shape[0]
        else:
            self.negative_data_weights = np.ones(negative_data.shape[0])

    def init_logger(self, name, debug):
        """
        Initialize loggers and log files

        Parameters
        ----------
        name : string
            name of directory for outputs
        """
        if not os.path.exists('./output/'+name+'/'):
            os.makedirs('./output/'+name+'/')
        self.logger = logging.getLogger(name)
        handler1 = logging.FileHandler('./output/'+name+'/full.log')
        handler1.setLevel(logging.DEBUG)
        self.logger.addHandler(handler1)
        handler2 = logging.FileHandler('./output/'+name+'/basic.log')
        handler2.setLevel(logging.INFO)
        self.logger.addHandler(handler2)

        handler3 = logging.StreamHandler(sys.stdout)
        if debug:
            handler3.setLevel(logging.DEBUG)
        else:
            handler3.setLevel(logging.INFO)

        #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        #handler3.setFormatter(formatter)
        self.logger.addHandler(handler3)
        self.logger.setLevel(logging.DEBUG)

        self.logger.critical('NEW RUN________________________________________')
        self.name = name
        self.plot_name = 0

    def initialize(self):
        """
        Initialize arrays with positive points and its utilities by finding a maximum of utility function.
        And initialize a first classifier.
        """
        result = self.optimize(lambda x: -self.utility(x), self.bounds)

        positive_data = (result[0]).reshape(1, -1)
        positive_data_utility = -1*result[1]

        pd = np.empty([1,len(self.bounds)])
        for i, b in enumerate(self.bounds):
            pd[0,i] = b[1] + 10* b[1]-b[0]
        pdu = 1

        self.p_classifier.initialize(self.negative_data, self.negative_data_weights, pd, pdu, self.logger)

        return positive_data, positive_data_utility

    def create_matrix(self, positive_data):
        """
        Creates initial game matrix by finding first classifier

        Parameters
        ----------
        positive_data : np.array
            Data to be classified (all labeled 0) - attackers points
        """
        matrix = np.array([self.p_classifier.classify_all(positive_data[-1,:])*self.utility(positive_data[-1,:])])
        print("matrix=", matrix)
        matrix = matrix.reshape(1,-1)
        print("matrix=", matrix)
        return matrix

    def actual_utility(self, point):
        """
        Calculates utility given by actual strategy of classifier

        Parameters
        ----------
        point : np.array
            Point to by evaluated
        """
        p_negative = self.p_classifier.classify(point)
        return p_negative*self.utility(point)

    def discretize(self, f, bounds, density):
        args = []
        for b in bounds:
            args.append(np.linspace(b[0], b[1], density))
        X = np.meshgrid(*args)
        X1 = list(map(lambda x: x[..., None], X))
        X1 = np.concatenate(X1, axis=-1)
        f_x = np.apply_along_axis(f, -1, X1)
        return (X, f_x)

    def find_attackers_br(self, positive_data):
        """
        Finds BR point to be played wrt actual classifiers strategy.

        """
        self.timer.start_measure("plot")
        if PLOT_ALL and len(self.bounds)==2:
            X, f_x = self.discretize(self.actual_utility, self.bounds, 100)
            ax = plt.axes(projection='3d')
            ax.plot_wireframe(X[0], X[1], f_x)
            plt.savefig('./output/'+self.name+'/'+str(self.plot_name)+'.png')
            plt.close()
            self.plot_name+=1
        elif PLOT_ALL and len(self.bounds)==1:
            t1 = np.arange(self.bounds[0][0], self.bounds[0][1], 0.1).reshape(-1, 1)
            plt.plot(t1, np.apply_along_axis(self.actual_utility, -1, t1))
            plt.savefig('./output/'+self.name+'/'+str(self.plot_name)+'.png')
            plt.close()
            self.plot_name+=1
        self.timer.end_measure("plot")

        self.timer.start_measure("BR")
        result = self.optimize(lambda x: -self.actual_utility(x), self.bounds)
        result2 = self.optimize2(lambda x: self.actual_utility(x), self.bounds, positive_data)

        self.logger.debug(f"PGD point = {result2[0]}")
        self.logger.debug(f"PGD value = {result2[1]}")

        self.timer.end_measure("BR")

        BR_point = result[0]
        BR_value = -1*result[1]

        return BR_point, BR_value

    def expand_matrix_attacker(self, matrix, positive_data):
        """
        Update game matrix by newly added attackers point

        Parameters
        ----------
        matrix : np.array
            Game matrix
        positive_data : np.array
            All attackers points
        """
        row = self.p_classifier.classify_all(positive_data[-1, :]) * self.utility(positive_data[-1, :])
        matrix = np.concatenate((matrix, row.reshape(1,-1)), axis=0)
        return matrix

    def expand_matrix_classifier(self, matrix, positive_data, positive_data_utility, column):
        """
        Update game matrix by newly added classifier

        Parameters
        ----------
        matrix : np.array
            Game matrix
        positive_data : np.array
            All attackers points
        positive_data_utility : np.array
            Utilities of all attackers points
        column : np.array
            new column of the matrix corresponding to the newly added classifier
        """
        column = column.reshape(-1, 1)
        if column.shape[0] < matrix.shape[0]:
            column = np.vstack((column, self.p_classifier.classify_last(positive_data[-1,:])*positive_data_utility[-1]))
        matrix = np.concatenate((matrix, column), axis=1)
        return matrix

    def expand_positive_data(self, positive_data, positive_data_utility, BR_point, BR_value, matrix, class_support):
        """
        Add BR point into the positive_data array, if point near to it is not there.

        Parameters
        ----------
        positive_data : np.array
            List of all attackers points
        positive_data_utility : np.array
            Utilities of all attackers points
        BR_point : np.array
            New point to be added
        BR_value : float
            Utility value of BR_point under actual strategies
        matrix : np.array
            Game matrix
        class_support : np.array
            Actual classifiers strategy
        """
        utilities = matrix.dot(class_support.reshape(-1, 1))
        self.logger.info(f"Points utilities = {utilities.T}")
        self.logger.info(f"new point utility = {BR_value}")
        self.logger.info(f"new point = {BR_point}")

        if is_higher_then_all_in_by(BR_value, utilities, UTILITY_STEP):
            self.logger.info(f"added new point = {BR_point}")
            positive_data = np.concatenate((positive_data, BR_point.reshape(1, -1)))
            positive_data_utility = np.append(positive_data_utility, self.utility(BR_point))
        else:
            self.logger.info(f"new point is not better than actual points, is not added = {BR_point}")
        return positive_data, positive_data_utility

    def update_history(self, history, support):
        """
        Saves the sctual strategy into the history for usage in mixed step

        Parameters
        ----------
        history : np.array
            history of few last restricted strategies
        support : np.array
            Actual strategy
        """
        history = np.roll(history, 1, axis=0)
        if history.shape[1] < support.shape[0]:
            history = np.c_[history, np.zeros(history.shape[0])]
        history[0,:] = support
        return history

    def mixing_step(self, matrix, positive_data, positive_data_utility, history_class, history_att):
        """
        One step of DO algorithm mixing between a few last strategies.
        With unset mixing_weights calculates a step of both players at once.

        Parameters
        ----------
        matrix : np.array
            Game matrix
        positive_data : np.array
            List of all attackers points
        positive_data_utility : np.array
            Utilities of all attackers points
        history_class : np.array(history_length, support_size)
            History of classifiers supports
        history_att : np.array(history_length, support_size)
            History of attacker supports
        """

        # solve restricted game
        self.logger.debug(f"Game matrix = {matrix}")
        self.timer.start_measure("nfg")

        if (self.fp_thresh is None):
            class_support, att_support, obj = self.solve_nfg(matrix)
        else:
            class_support, att_support, obj = self.solve_nfg(matrix, self.p_classifier.get_fp(), self.fp_thresh)

        # update history and compute mixed strategy
        if self.mixing_weights.shape[0] > 0:
            history_class = self.update_history(history_class, class_support)
            history_att = self.update_history(history_att, att_support)

            class_support = self.mixing_weights.dot(history_class)
            att_support = self.mixing_weights.dot(history_att)

        self.p_classifier.update_support(class_support)
        self.timer.end_measure("nfg")

        self.logger.debug(f"attacker data: {positive_data.T}")
        self.logger.debug(f"attacker support: {att_support}")
        self.logger.debug(f"defender support: {class_support}")
        self.logger.debug(f"classifires probs: {self.p_classifier.probabilities}")

        if PLOT_ALL:
            self.plot(positive_data, self.negative_data, att_support, utility=self.utility)

        # find ATTACKERS BR
        self.logger.info(f"finding best response for attacker....")
        BR_point, BR_value = self.find_attackers_br(positive_data.T)
        self.logger.info(f"attackers best response search ended")

        self.logger.debug(f"BR point: {BR_point}")
        self.logger.debug(f"BR value: {BR_value}")

        # expand game matrix attacker
        self.timer.start_measure("update_matrix")
        self.logger.info(f"updating game matrix for attacker...")
        positive_data_new, positive_data_utility_new = self.expand_positive_data(positive_data, positive_data_utility,
                                                                         BR_point, BR_value, matrix, class_support)

        if np.size(positive_data, 0) != np.size(positive_data_new, 0):
            matrix = self.expand_matrix_attacker(matrix, positive_data_new)
            self.logger.info(f"expanding game rows, new better attacker point found...")
        else:
            self.logger.info(f"game matrix stays same, no better attacker point found...")
        self.timer.end_measure("update_matrix")

        # find BR classifier, and check if is present
        self.timer.start_measure("train_new")
        self.logger.info(f"finding best response for defender....")
        column = self.p_classifier.update(positive_data, positive_data_utility, att_support, matrix)
        self.logger.info(f"ending training classifiers")

        self.timer.end_measure("train_new")
        self.logger.info(f"defender best response search ended")
        self.logger.debug(f"classifires probs: {self.p_classifier.probabilities}")

        # expand game matrix
        self.timer.start_measure("update_matrix")
        self.logger.info(f"updating game matrix for defender...")

        if not column is None:#len_classifier != len(self.p_classifier):
            matrix = self.expand_matrix_classifier(matrix, positive_data_new, positive_data_utility_new, column)
            self.logger.debug(f"matrix classifier = {matrix}")
            self.logger.info(f"expanding game columns, new better classifier found...")
        else:
            self.logger.info(f"game matrix stays same, no better classifier found...")

        if len(self.p_classifier) > class_support.shape[0]:
            class_support = np.hstack((class_support, 0))

        self.timer.end_measure("update_matrix")
        self.logger.debug(f"Game matrix = {matrix}")

        if (np.size(positive_data, 0) == np.size(positive_data_new, 0) and (column is None)):
            self.logger.info(f"endling, no new best responses found, double oracle should have converged!")
            return matrix, positive_data_new, positive_data_utility_new, history_class, history_att, (att_support, obj), True

        return matrix, positive_data_new, positive_data_utility_new, history_class, history_att, (att_support, obj), False

    def alternating_step(self, matrix, positive_data, positive_data_utility, history_class, history_att):
        """
        One step of DO algorithm calculates an alternating step for both players.

        Parameters
        ----------
        matrix : np.array
            Game matrix
        positive_data : np.array
            List of all attackers points
        positive_data_utility : np.array
            Utilities of all attackers points
        history_class : np.array(history_length, support_size)
            History of classifiers supports
        history_att : np.array(history_length, support_size)
            History of attacker supports
        """
        # solve nfg
        self.timer.start_measure("nfg")
        if (self.fp_thresh is None):
            class_support, att_support, obj = self.solve_nfg(matrix)
        else:
            class_support, att_support, obj = self.solve_nfg(matrix, self.p_classifier.get_fp(), self.fp_thresh)
        self.timer.end_measure("nfg")
        self.logger.debug(f"positive data: {positive_data}")
        self.logger.debug(f"positive support: {att_support}")
        if PLOT_ALL:
            self.p_classifier.update_support(class_support)
            self.plot(positive_data, self.negative_data, att_support, utility=self.utility)

        # find BR classifier, and check if is present
        self.timer.start_measure("train_new")
        column = self.p_classifier.update(positive_data, positive_data_utility, att_support, matrix)
        self.timer.end_measure("train_new")
        self.logger.debug(f"classifires probs: {self.p_classifier.probabilities}")

        if column is None:
            self.logger.info('No classifier added')
            return matrix, positive_data, positive_data_utility, history_class, history_att, (att_support, obj), True

        # expand game matrix
        self.timer.start_measure("update_matrix")
        matrix = self.expand_matrix_classifier(matrix, positive_data, positive_data_utility, column)
        self.timer.end_measure("update_matrix")

        # solve game
        self.timer.start_measure("nfg")
        if (self.fp_thresh is None):
            class_support, att_support, obj = self.solve_nfg(matrix)
        else:
            class_support, att_support, obj = self.solve_nfg(matrix, self.p_classifier.get_fp(), self.fp_thresh)
        self.timer.end_measure("nfg")
        self.p_classifier.update_support(class_support)
        self.logger.debug(f"classifires probs: {self.p_classifier.probabilities}")

        if PLOT_ALL:
            self.plot(positive_data, self.negative_data, att_support, utility=self.utility)

        # attacker finds BR
        self.timer.start_measure("BR")
        BR_point, BR_value = self.find_attackers_br()
        self.timer.end_measure("BR")

        self.logger.debug(f"BR point: {BR_point}")
        self.logger.debug(f"BR value: {BR_value}")

        # updates attackers lists, check if similar BR is present
        self.timer.start_measure("update_matrix")
        num_positive = np.size(positive_data, 0)
        positive_data, positive_data_utility = self.expand_positive_data(positive_data, positive_data_utility,
                                                                         BR_point, BR_value, matrix, class_support)
        if (num_positive == np.size(positive_data, 0)):
            self.logger.info('No point added')
            return matrix, positive_data, positive_data_utility, history_class, history_att, (att_support, obj), True

        # expand game matrix
        matrix = self.expand_matrix_attacker(matrix, positive_data)
        self.timer.end_measure("update_matrix")

        return matrix, positive_data, positive_data_utility, history_class, history_att, (att_support, obj), False

    def compute(self, max_iters, verbose=True):
        """
        Solves the problem by DO algorithm

        Parameters
        ----------
        max_iters : int
            limitation on the maximum iterations
        verbose: bool
            print extended info every iteration
        """
        self.timer.start_measure("init")
        positive_data, positive_data_utility = self.initialize()
        self.logger.debug(f"positive data: {positive_data}")
        self.logger.debug(f"positive utility: {positive_data_utility}")

        matrix = self.create_matrix(positive_data)

        history_class = np.ones((self.mixing_weights.shape[0],1))
        history_att = np.ones((self.mixing_weights.shape[0], 1))
        self.timer.end_measure("init")

        final = (0, 0)

        for i in range(max_iters):
            self.logger.info(f"---------------------------- iteration: {i}")

            self.logger.info(f"Size of game = {matrix.shape}")
            self.logger.info(f"Objective = {final[1]}")
            self.logger.info(f"Support of attacker = {np.sum(final[0] > THETA)}")
            self.logger.info(f"Support of classifier = {self.p_classifier.support_size()}")
            self.logger.info(f"init time = {self.timer.get_measurement('init')}")
            self.logger.info(f"NFG time = {self.timer.get_measurement('nfg')}")
            self.logger.info(f"plot time = {self.timer.get_measurement('plot')}")
            self.logger.info(f"BR time = {self.timer.get_measurement('BR')}")
            self.logger.info(f"Train time = {self.timer.get_measurement('train_new')}")
            self.logger.info(f"Matrix update time = {self.timer.get_measurement('update_matrix')}")

            matrix, positive_data, positive_data_utility, history_class, history_att, final, end = \
                self.step(matrix, positive_data, positive_data_utility, history_class, history_att)

            if end:
                break

        self.logger.info(f"----------------------------")
        self.logger.warning(f"Size of game = {matrix.shape}")
        self.logger.warning(f"Objective = {final[1]}")
        self.logger.warning(f"Number of iterations = {i+1}")
        self.logger.warning(f"Support of attacker = {np.sum(final[0]>THETA)}")
        self.logger.warning(f"Support of classifier = {self.p_classifier.support_size()}")
        self.logger.info(f"init time = {self.timer.get_measurement('init')}")
        self.logger.info(f"NFG time = {self.timer.get_measurement('nfg')}")
        self.logger.info(f"plot time = {self.timer.get_measurement('plot')}")
        self.logger.info(f"BR time = {self.timer.get_measurement('BR')}")
        self.logger.info(f"Train time = {self.timer.get_measurement('train_new')}")
        self.logger.info(f"Matrix update time = {self.timer.get_measurement('update_matrix')}")

        if end:
            if len(self.bounds) == 2:
                X, f_x = self.discretize(self.actual_utility, self.bounds, 100)
                ax = plt.axes(projection='3d')
                ax.plot_wireframe(X[0], X[1], f_x)
                plt.savefig('./output/' + self.name + '/' + str(self.plot_name) + '.png')
                plt.close()
                self.plot_name += 1
            self.plot(positive_data, self.negative_data, final[0], self.utility)
        else:
            self.logger.warning(f"Max iteration reached!")

    def plot(self, positive_data, negative_data, att_support=0, utility=lambda x: x[0]):
        """
        Plots the results

        Parameters
        ----------
        positive_data : np.array
            List of all attackers points
        negwtive_data : np.array
            List of all benign points
        att_support : np.array
            Actual attackers strategy
        utility : function(np.array(d)) -> float
            utility function callable on point
        """
        if positive_data.shape[1] == 1:
            self.plot1d(positive_data, negative_data, att_support, utility)
        elif positive_data.shape[1] == 2:
            self.plot2d(positive_data, negative_data, att_support, utility)

    def plot2d(self, positive_data, negative_data, att_support, utility=lambda x: x[0]+x[1]):
        """
        Plots the results when the space is 2D

        Parameters
        ----------
        positive_data : np.array
            List of all attackers points
        negwtive_data : np.array
            List of all benign points
        att_support : np.array
            Actual attackers strategy
        utility : function(np.array(d)) -> float
            utility function callable on point
        """
        x_min = self.bounds[0][0]
        x_max = self.bounds[0][1]
        y_min = self.bounds[1][0]
        y_max = self.bounds[1][1]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(positive_data[att_support < THETA, 0], positive_data[att_support <THETA, 1],
                   att_support[att_support < THETA], c='r')
        ax.scatter(positive_data[att_support >= THETA, 0], positive_data[att_support >= THETA, 1],
                   att_support[att_support >= THETA], c='m')
        ax.scatter(positive_data[-1, 0], positive_data[-1, 1], att_support[-1], c='y')
        ax.scatter(negative_data[:, 0], negative_data[:, 1], 1, c='g')

        ax.set_xlim3d(x_min, x_max)
        ax.set_ylim3d(y_min, y_max)
        ax.set_zlim3d(0, 1)

        for i, clf in enumerate(self.p_classifier.classifiers):
            XX, YY = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]

            classification = self.p_classifier.classify_by_one_classifier(np.c_[XX.ravel(), YY.ravel()], clf)

            act_utility = classification

            # Put the result into a color plot
            act_utility = act_utility.reshape(XX.shape)
            classification = (classification.reshape(XX.shape) - 0.5) + self.p_classifier.probabilities[i]

            color = ['g']
            if i == len(self.p_classifier)-1:
                color = ['y']

            if (self.p_classifier.probabilities[i] < THETA):
                plt.contour(XX, YY, classification, colors=['b'], linestyles=['-'],
                            levels=[self.p_classifier.probabilities[i]])
            else:
                plt.contour(XX, YY, classification, colors=color, linestyles=['-'],
                            levels=[self.p_classifier.probabilities[i]])

        plt.savefig('./output/' + self.name + '/' + str(self.plot_name) + '.png')
        plt.close()
        self.plot_name += 1

    def plot1d(self, positive_data, negative_data, att_support=0, utility=lambda x: x[0]):
        """
        Plots the results when the space is 1D

        Parameters
        ----------
        positive_data : np.array
            List of all attackers points
        negwtive_data : np.array
            List of all benign points
        att_support : np.array
            Actual attackers strategy
        utility : function(np.array(d)) -> float
            utility function callable on point
        """
        x_min = self.bounds[0][0]
        x_max = self.bounds[0][1]

        fig = plt.figure()

        plt.xlim(x_min, x_max)
        plt.ylim(0, 1)

        plt.scatter(positive_data[att_support < THETA, 0], att_support[att_support < THETA], c='r')
        plt.scatter(positive_data[att_support >= THETA, 0], att_support[att_support >= THETA], c='m')
        plt.scatter(positive_data[-1, 0], att_support[-1], c='y')
        plt.scatter(negative_data[:, 0], np.ones(negative_data[:, 0].shape), c='g')

        for i, clf in enumerate(self.p_classifier.classifiers):
            t1 = np.arange(x_min, x_max, 0.1)

            classification = self.p_classifier.classify_by_one_classifier(t1.reshape(-1,1), clf)

            if(np.any(classification<1)):
                plt.scatter(x_min+0.1*np.where(classification<1)[0][0], self.p_classifier.probabilities[i], marker="|")

        plt.savefig('./output/' + self.name + '/' + str(self.plot_name) + '.png')
        plt.close()
        self.plot_name += 1