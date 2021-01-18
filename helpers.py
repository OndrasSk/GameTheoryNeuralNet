import numpy as np

def equal_in_tolerance(a, b, theta):
    """
    Check if points a and b are closer than theta

    Parameters
    ----------
    a : np.array
        first point
    b : np.array
        second point
    theta : float
        difference threshold
    """
    return np.linalg.norm(a-b) < theta

def is_higher_then_all_in_by(a, b, theta):
    """
    Compares if all elements in a are higher than elements in b on corresponding places by theta

    Parameters
    ----------
    a : np.array
        first array of numbers
    b : np.array
        second array of number
    theta : float
        difference threshold
    """
    return np.all(a-b>theta)