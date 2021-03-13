"""
Given a vector of data, compute the softmax scores of the vector.

For a review of softmax, check out: https://en.wikipedia.org/wiki/Softmax_function
"""

import numpy as np

VEC = [-0.25560104, 0.06393334, -0.43760861, 0.35258494, -0.06174621]


def softmax(vec):
    """
    Compute the softmax of the vector
    :param vec: Vector represented as a numpy array
    :return: The softmax scores of the provided vector
    """
    sm = np.exp(vec) / np.sum(np.exp(vec))

    return sm


softmax(np.array(VEC)).tolist()
