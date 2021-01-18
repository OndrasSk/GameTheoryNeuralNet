from abc import ABC, abstractmethod

import numpy as np
#from experiments.time_measure import timer
from timer import Timer

class OptimizationInterface(ABC):
    def __init__(self, p_classifier, negative_points, n_points, logger):
        self.timer = Timer()
        self.logger = logger

        self.negative_data = negative_points
        self.p_classifier = p_classifier
        self.n_points = n_points
        self.f = np.inf
        self.x = []
        self.X1 = None

    @abstractmethod
    def optimize(self, fun, bounds):
        """
        Find a minimum of function in bounds

        Parameters
        ----------
        fun : function(np.array(d)) -> float
            function callable on point
        bounds : ((lower_bound_1, upper_dound_1), ..., (lower_bound_d, upper_dound_d))
            bounds of game space

        Returns
        -------
        np.array(d)
            point with minimal value
        float
            minimal value
        """
        pass