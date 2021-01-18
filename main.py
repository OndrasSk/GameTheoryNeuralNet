import os
import sys

import numpy as np

from classifiers.probabilisticDecisionTree import ProbabilisticDecisionTree
from classifiers.probabilisticNN_basic import ProbabilisticNN
from classifiers.probabilisticSVM import ProbabilisticSVM
from discretizationSolver import DiscretizationSolver
from doubleOracle import DoubleOracle
from optimization.optimizationBasinhopping import OptimizationBasinhopping
from optimization.optimizationBasinhoppingDiscretization import OptimizationBasinhoppingDiscretization
from optimization.optimizationDiscretization import OptimizationDiscretization
from utilityFunctions import linear_utility, one_maximum, two_maxima

import matplotlib
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('Agg')

SHOW = False


def discretize(f, bounds, density):
    args = []
    for b in bounds:
        args.append(np.linspace(b[0], b[1], density))
    X = np.meshgrid(*args)
    X1 = list(map(lambda x: x[..., None], X))
    X1 = np.concatenate(X1, axis=-1)
    f_x = np.apply_along_axis(f, -1, X1)
    return (X, f_x)


def plot_2d(f, bounds, name, points=None):
    X, f_x = discretize(f, bounds, 100)
    ax = plt.axes(projection='3d')
    if not points is None:
        ax.scatter(points[:, 0], points[:, 1], np.ones(points[:, 1].shape) * 2, c='g')
    ax.plot_wireframe(X[0], X[1], f_x)

    if SHOW:
        plt.show()
    else:
        if not os.path.exists('./output/' + name + '/'):
            os.makedirs('./output/' + name + '/')
        plt.savefig('./output/' + name + '/_0.png')
        plt.close()


def plot_1d(f, bounds, name, points=None):
    t1 = np.arange(bounds[0][0], bounds[0][1], 0.1).reshape(-1, 1)
    plt.plot(t1, np.apply_along_axis(f, -1, t1))
    if not points is None:
        plt.scatter(points, np.ones(points.shape) * 0.1, c='g')
    if SHOW:
        plt.show()
    else:
        if not os.path.exists('./output/' + name + '/'):
            os.makedirs('./output/' + name + '/')
        plt.savefig('./output/' + name + '/_0.png')
        plt.close()


def print_help():
    print("""
    
    Usage of the script:
    
python main.py function points fp_threshold algorithm *params
        [optimizer] [step] [weights]

function:       0 -> Linear utility
                1 -> Utility with one maximum
                2 -> Utility with two maxima
            
points:         a path to the *.npy file with the benign points

fp_threshold:   float - a number between 0 and 1 to limit
                        a false-positive rate
                None  - algorithm expects classifier
                        with hard false-positive constraint
                
algorithm:      discretization -> discretization algorithm
                SVM            -> Double Oracle with SVM classifier
                NN             -> Double Oracle with neural network
                DT             -> Double Oracle with decision tree

optimizer:      0 -> discretization optimizer
                1 -> Basin-Hopping optimizer with discretization
                2 -> Basin-Hopping optimizer
                
step:           0 -> simultaneous computation of the attacker's BR
                1 -> alternating computation of the attacker's BR
                2 -> simultaneous computation of the attacker's BR
                        on weighted a few strategies in history
                
weights:        0 -> benign points has weight 1
                1 -> benign points has weight 1/n
                
                
                
*params:        depends on the algorithm settings

    discretization:
        density: int  - density of sampling
                 None - exact computation
        
    SVM:
        degree: int - degree of polynomial kernel
        
    NN:
        hidden_size:    int   - number of neurons in hidden layer
        epochs:         int   - number of epochs before check of
                                classifier utility
        iterations:     int   - number of repetitions of training
                                and utility checks
        last for init:  bool  - initialize the NN with weights
                                from previous training
        gradient:       float - required descent of loss,
                        None  - the addition of the first better
        
    DT:
        max depth:  int   - a maximal depth of the tree
                    None  - unlimited
        gradient:   float - required descent of weighted
                            misclassification change,
                    None  - unlimited
                    First - the addition of the first better
    """)
    exit()


def get_rest(i):
    steps = ["atonce", "alternating", "mixing"]
    optimizers = [OptimizationDiscretization, OptimizationBasinhoppingDiscretization, OptimizationBasinhopping]
    weight_options = ["ones", "uniform"]

    optimize = optimizers[0]
    step = steps[0]
    weights = weight_options[0]

    if len(sys.argv) > i:
        optimize = optimizers[int(sys.argv[i])]
    if len(sys.argv) > i + 1:
        step = steps[int(sys.argv[i + 1])]
    if len(sys.argv) > i + 2:
        weights = weight_options[int(sys.argv[i + 2])]

    return optimize, step, weights


def example_universal(data, function, probabilistic_classifier, bounds, fp_thresh, optimizer, step,
                      weights, name, debug):
    if len(bounds) == 2:
        plot_2d(function, bounds, name)
    elif len(bounds) == 1:
        plot_1d(function, bounds, name)

    double_oracle = DoubleOracle(negative_data=data, utility=function,
                                 probabilistic_classifier=probabilistic_classifier,
                                 bounds=bounds, fp_thresh=fp_thresh, optimizer=optimizer, step_type=step,
                                 negative_data_weights_generation=weights, name=name, debug=debug)

    double_oracle.compute(9000, True)


if __name__ == "__main__":
    functions = [linear_utility, one_maximum, two_maxima]

    if len(sys.argv) < 6 or sys.argv[1] == "help":
        print_help()

    f = functions[int(sys.argv[1])]
    points = np.load(sys.argv[2])
    bounds = [(0, 10)] * points.shape[1]

    if sys.argv[3] == "None":
        fp_thresh = None
    else:
        fp_thresh = float(sys.argv[3])

    name = '_'.join(sys.argv[1:])

    if sys.argv[4] == "discretization":
        solver = DiscretizationSolver(points, f, bounds, fp_thresh)
        if sys.argv[5] == "None":
            density = None
        else:
            density = int(sys.argv[5])
        solver.compute(density)
    else:
        if sys.argv[4] == "SVM":
            p_classifier = ProbabilisticSVM('poly', C=10000.0, degree=int(sys.argv[5]))
            optimize, step, weights = get_rest(6)
        elif sys.argv[4] == "NN":
            if len(sys.argv) < 9:
                print_help()
            hidden_size = int(sys.argv[5])
            epochs = int(sys.argv[6])
            iterations = int(sys.argv[7])
            cont_learning = bool(sys.argv[8] != "False")
            if sys.argv[9] == "None":
                gradient = np.inf
            else:
                gradient = float(sys.argv[9])
            p_classifier = ProbabilisticNN(epochs=epochs, iterations=iterations, hidden_size=hidden_size,
                                           continue_training=cont_learning, gradient=gradient)
            optimize, step, weights = get_rest(10)
        elif sys.argv[4] == "DT":
            if len(sys.argv) < 7:
                print_help()
            if sys.argv[5] == "None":
                md = None
            else:
                md = int(sys.argv[5])
            if sys.argv[6] == "None":
                gradient = None
            elif sys.argv[6] == "First":
                gradient = np.inf
            else:
                gradient = float(sys.argv[6])
            pr_classifier = ProbabilisticDecisionTree(max_depth=md, gradient=gradient)
            optimize, step, weights = get_rest(7)
        example_universal(points, f, p_classifier, bounds, fp_thresh, optimize, step, weights, name)
