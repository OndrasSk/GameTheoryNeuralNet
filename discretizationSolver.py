import warnings
from cvxopt import matrix, solvers

import numpy as np

class DiscretizationSolver:
    def __init__(self, negative_data, utility, bounds, fp_thresh=None):
        """
        Initialize discretization solver

        Parameters
        ----------
        negative_data : np.array
            data of negative class (n, d)
        utility : function(np.array(d)) -> float
            utility function callable on point
        bounds : ((lower_bound_1, upper_dound_1), ..., (lower_bound_d, upper_dound_d))
            bounds of game space
        fp_thresh : float
            threshold for the false-positive constraint
        """
        self.negative_data = negative_data
        self.utility = utility
        self.bounds = bounds

        if fp_thresh is None:
            warnings.warn('It is not possible use hard FP constrain. Soft FP constraint 0.01 forced.', Warning)
            fp_thresh = 0.01

        self.fp_thresh = fp_thresh

    def discretize(self, density):
        """
        Discretize the space

        Parameters
        ----------
        density : int
            Density for discretization
        """
        args = []
        for b in self.bounds:
            h = (b[1]-b[0])/(2*density)
            args.append(np.linspace(b[0]+h, b[1]-h, density))
        X = np.meshgrid(*args)
        X1 = list(map(lambda x: x[..., None], X))
        X1 = np.concatenate(X1, axis=-1)
        f_x = np.apply_along_axis(self.utility, -1, X1)
        return (X, f_x)

    def solve_cvx(self, f_x, fp, fp_thresh=0.01):
        """
        Solve discretized game LP CVXOPT

        Parameters
        ----------
        f_x : np.array
            utility diskrétních intervalů
        fp : np.array
            list of false-positive rates of all classifiers
        fp_thresh : float
            threshold for the false-positive constraint
        """
        solvers.options['show_progress'] = False

        c = np.zeros((f_x.shape[0]+1,1))
        c[-1] = 1

        A = np.hstack((fp,0))
        r = np.hstack((-1*np.diag(f_x), -1*np.ones((f_x.shape[0], 1))))
        A = np.vstack((A,r))
        v = np.hstack((np.diag(np.ones(f_x.shape[0])), np.zeros((f_x.shape[0], 1))))
        A = np.vstack((A,-1*v))
        A = np.vstack((A, v))

        b = np.vstack((fp_thresh, -1* f_x.reshape((-1,1)), np.zeros((f_x.shape[0], 1)), np.ones((f_x.shape[0], 1))))

        A1 = matrix(A)
        b = matrix(b)
        c = matrix(c)

        sol = solvers.lp(c, A1, b)
        solution1 = np.reshape(sol['x'][:-1, :], (-1))
        solution2 = np.reshape(sol['z'][1:f_x.shape[0]+1, :], (-1))

        return solution1, solution2, sol['primal objective']

    def compute(self, density=None):
        """
        Discretize the space. If density is None, algorithm computes the exact value.

        Parameters
        ----------
        density : int or None
            Density for discretization
        """
        if density is None:
            f_x = np.apply_along_axis(self.utility, -1, self.negative_data)
            fp = np.ones(f_x.shape) / self.negative_data.shape[0]
        else:
            X, f_x = self.discretize(density)
            fp = np.histogramdd(self.negative_data, range=self.bounds, bins=density)[0]/self.negative_data.shape[0]
            fp = fp.reshape(1,-1)
            f_x = f_x.reshape(1,-1)
            f_x = f_x[fp !=0]
            fp = fp[fp != 0]

        (solution, solution2, val) = self.solve_cvx(f_x, fp, fp_thresh=self.fp_thresh)
        print("Final value:", val)


