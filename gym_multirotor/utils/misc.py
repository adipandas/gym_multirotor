import numpy as np


def get_magnitude(vector):
    """
    Get the magnitude of input vector

    :param vector: input vector of shape (n,)
    :type vector: numpy.ndarray
    :return: magnitude of the input vector
    :rtype: float
    """

    mag = np.sqrt(np.sum(np.square(vector)))
    return mag
