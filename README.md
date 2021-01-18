# GameTheoryNeuralNetwork

Classifiers are used in security to classify the actions of an adversary as malicious or benign. This interaction can be modeled as a game; the strategy of one player corresponds to setting parameters of a classifier, the strategy of the opponent is to choose such an input that causes misclassification. Recently, there has been a large volume of work on verification of classifiers and generating adversarial samples. However, it is not clear whether these approaches are compatible with reward functions and strategy constraints the attacker typically has. The goal of this double-oracle framework is to provide experimental analysis and identify advantages and disadvantages of existing verification methods and methods for generating adversarial samples, which are compatible with reward functions and strategy constraints.

# Simple usage:
    
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
