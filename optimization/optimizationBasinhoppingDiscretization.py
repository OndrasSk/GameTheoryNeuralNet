import threading

import numpy as np
import scipy.optimize as opt
from scipy import sparse
from optimization.optimizationInterface import OptimizationInterface


class OptimizationBasinhoppingDiscretization(OptimizationInterface):

    def __init__(self, p_classifier, negative_points, n_points, logger):
        super().__init__(p_classifier, negative_points, n_points, logger)

    def expand_classifyed(self):
        if self.classified.shape[1]<len(self.p_classifier):
            row = np.apply_along_axis(self.p_classifier.classify_last, -1, self.X1)
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
            args = []
            for b in bounds:
                args.append(np.linspace(b[0], b[1], 100))
            X = np.meshgrid(*args)
            X1 = list(map(lambda x: x[..., None], X))
            self.X1 = np.concatenate(X1, axis=-1).reshape([-1, len(bounds)])
            self.utilities = np.apply_along_axis(fun, -1, self.X1)
            self.classified = sparse.csr_matrix((self.utilities.shape[0], 0))  # np.empty((0,self.utilities.shape[0]))
            self.timer.end_measure("init")

        self.timer.start_measure("expand")
        self.expand_classifyed()
        self.timer.end_measure("expand")

        self.timer.start_measure("points")
        self.lock = threading.Lock()
        points = self.best_points()
        self.timer.end_measure("points")

        max = self.f

        self.timer.start_measure("threads")

        thr = []
        for point in points:
            thr.append(Basinhopping_improved(self, point, fun, bounds))
            thr[-1].start()

        for t in thr:
            t.join()
        ret = self.x
        ret_value = self.f
        self.f = np.inf
        self.x = []
        self.timer.end_measure("threads")

        self.logger.info(f"BH init time = {self.timer.get_measurement('init')}")
        self.logger.info(f"BH expand time = {self.timer.get_measurement('expand')}")
        self.logger.info(f"BH points time = {self.timer.get_measurement('points')}")
        self.logger.info(f"BH threads time = {self.timer.get_measurement('one_run')}")
        self.logger.info(f"BH threads time = {self.timer.get_measurement('threads')}")

        return ret, ret_value


class Basinhopping_improved(threading.Thread):
    def __init__(self, opt, point, fun, bounds):
        ''' Constructor. '''

        threading.Thread.__init__(self)
        self.iter = 0
        self.init_point = point
        self.fun = fun
        self.bounds = bounds

        self.opt = opt

    def run(self):
        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": self.bounds}
        opt.basinhopping(self.fun, self.init_point, minimizer_kwargs=minimizer_kwargs, callback=self.callback, niter=50000).x

    def callback(self, x, f, accept):
        self.opt.lock.acquire()
        if f < self.opt.f:
            self.iter = -1
            self.opt.f = f
            self.opt.x = x
        self.opt.lock.release()

        self.iter += 1
        if self.iter < 10:
            return False
        else:
            return True