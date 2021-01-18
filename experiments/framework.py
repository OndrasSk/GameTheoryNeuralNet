import numpy as np
import sys
from pathlib import Path

from classifiers.probabilisticNN_basic import ProbabilisticNN
from main import example_universal
from optimization.optimizationBasinhopping import OptimizationBasinhopping
from optimization.optimizationBasinhoppingDiscretization import OptimizationBasinhoppingDiscretization
from optimization.optimizationDiscretization import OptimizationDiscretization
from utilityFunctions import linear_utility, one_maximum, two_maxima


def neuralNetworkExperiment(function, data_path, fp_thresh, algorithm, optimizer, step,
                            weights, hidden_size, epochs, iterations, cont_learning, gradient, debug):
    functions = [linear_utility, one_maximum, two_maxima]
    f = functions[function]
    levels_up = 1
    path = str(Path().absolute().parents[levels_up - 1])
    path += "/data/" + data_path
    points = np.load(path)
    bounds = [(0, 10)] * points.shape[1]
    algorithm = "NN"

    p_classifier = ProbabilisticNN(epochs=epochs, iterations=iterations, hidden_size=hidden_size,
                                   continue_training=cont_learning, gradient=gradient)

    steps = ["atonce", "alternating", "mixing"]
    optimizers = [OptimizationDiscretization, OptimizationBasinhoppingDiscretization, OptimizationBasinhopping]
    weight_options = ["ones", "uniform"]

    step = steps[step]
    optimize = optimizers[optimizer]
    weights = weight_options[weights]

    name = '_'.join([str(elem) for elem in [data_path, fp_thresh, algorithm, hidden_size, epochs, iterations,
                                            cont_learning, gradient]])

    name = str(function) + "_data/" + name

    example_universal(points, f, p_classifier, bounds, fp_thresh, optimize, step, weights, name, debug)


if __name__ == "__main__":
    # 'Classic' NN test setup reverse engineered from original author Prokop Silhavy diploma thesis experiments
    #neuralNetworkExperiment(0, "data0_dim1.npy", 0.01, "NN", 0, 0, 0, 10, 1000, 100, False, 0.01, True)

    neuralNetworkExperiment(0, "data0_dim1.npy", 0.01, "NN", 0, 0, 0, 10, 1000, 100, False, 0.01, True)

    """
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
    """