import numpy as np

def linear_utility(x):
    return (1/x.shape[0]) * np.sum(x)

def one_maximum(x):
    x = (x - 5)**2
    return 10 - ( np.sum(x)/(x.shape[0]*5/2))

def two_maxima(x):
    x = (x - 5)*x.shape[-1]
    res = np.sum(x**4, axis=-1)
    res -= (25 * x.shape[-1]) * np.sum(x, axis=-1)**2
    res = -res /(625/40 * (x.shape[-1]**5))

    if res < 0:
        return 0
    return res