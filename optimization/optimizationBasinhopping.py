from abc import abstractmethod

import scipy.optimize as opt
from optimization.optimizationInterface import OptimizationInterface


class OptimizationBasinhopping(OptimizationInterface):

    def __init__(self, p_classifier, negative_points, n_points, logger):
        super().__init__(p_classifier, negative_points, n_points, logger)

    def optimize(self, fun, bounds):
        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}
        res = opt.basinhopping(fun, self.negative_data[0,:], minimizer_kwargs=minimizer_kwargs, niter=5000)
        ret = res.x
        ret_value = res.fun

        return ret, ret_value