import numpy as np
import scipy.optimize as opt
from scipy import sparse

from optimization.optimizationInterface import OptimizationInterface


class OptimizationDiscretization(OptimizationInterface):
    def __init__(self, p_classifier, negative_points, n_points, logger):
        super().__init__(p_classifier, negative_points, n_points, logger)

    def expand_classifyed(self):
        while self.classified.shape[1]<len(self.p_classifier):
            row = np.float64(np.apply_along_axis(lambda x: self.p_classifier.classify_by_one_classifier(x.reshape(1, -1),
                                            self.p_classifier.classifiers[self.classified.shape[1]]), 1, self.X1))
            self.classified = sparse.hstack([self.classified, row])

    def best_points(self):
        if self.classified.shape[1] > 0:
            probabilities = self.classified.dot(self.p_classifier.probabilities)
            f_x =  np.multiply(self.utilities, probabilities)
        else:
            f_x = self.utilities

        idxs = np.argpartition(f_x, self.n_points)[:self.n_points]

        best = np.argmin(f_x[idxs])
        self.f = f_x[idxs][best]
        self.x = self.X1[idxs,:][best,:]

        return self.X1[idxs,:]

    def optimize(self, fun, bounds):

        if self.X1 is None:
            self.timer.start_measure("init")
            self.negative_utility = np.apply_along_axis(fun, -1, self.negative_data)

            args = []
            for b in bounds:
                args.append(np.linspace(b[0], b[1], 100))
            X = np.meshgrid(*args)
            X1 = list(map(lambda x: x[..., None], X))
            self.X1 = np.concatenate(X1, axis=-1).reshape([-1,len(bounds)])
            self.utilities = np.apply_along_axis(fun, -1, self.X1)
            self.classified = sparse.csr_matrix((self.utilities.shape[0], 0))#np.empty((0,self.utilities.shape[0]))
            self.timer.end_measure("init")

        self.timer.start_measure("expand")
        self.expand_classifyed()
        self.timer.end_measure("expand")

        self.timer.start_measure("points")
        points = self.best_points()
        self.timer.end_measure("points")

        self.timer.start_measure("n_points")
        ok = self.negative_utility<self.f
        if np.any(ok):
            utils = np.apply_along_axis(fun, -1, self.negative_data[ok,:])
            i = np.argmin(utils)
            self.f = utils[i]
            self.x = self.negative_data[ok,:][i,:]

        self.timer.end_measure("n_points")

        self.timer.start_measure("one_run")
        for point in points:
            sol = opt.minimize(fun, point, method="L-BFGS-B", bounds=bounds)
            f = fun(sol.x)
            if f < self.f:
                self.f = f
                self.x = sol.x
        self.timer.end_measure("one_run")

        ret = self.x
        ret_value = self.f
        self.f = np.inf
        self.x = []

        self.logger.info(f"BH init time = {self.timer.get_measurement('init')}")
        self.logger.info(f"BH expand time = {self.timer.get_measurement('expand')}")
        self.logger.info(f"BH points time = {self.timer.get_measurement('points')}")
        self.logger.info(f"BH n_points time = {self.timer.get_measurement('n_points')}")
        self.logger.info(f"BH threads time = {self.timer.get_measurement('one_run')}")

        return ret, ret_value