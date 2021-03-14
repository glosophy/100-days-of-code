"""
Given a vector of data, normalize the vector to have 0 mean and variance of 1.

This is a commonly used normalization technique in various AI domains including computer
vision. It is often referred to as Z-score normalization.
"""

import numpy as np

VEC = [-0.25560104, 0.06393334, -0.43760861, 0.35258494, -0.06174621]


def normalize(vec):
    """
    Normalize the vector to have 0 mean and unit variance.
    :param vec1: Vector represented as a numpy array
    :return: The normalized vector
    """
    med = vec - np.mean(vec)
    std = med / np.std(med)

    return std


normalize(np.array(VEC)).tolist()
