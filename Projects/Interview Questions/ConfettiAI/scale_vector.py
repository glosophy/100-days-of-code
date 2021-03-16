"""
Given a vector of data, write a function to scale (multiply) all the entries in the vector
by the mean of the entries.
"""

import numpy as np

VEC = [-0.25560104, 0.06393334, -0.43760861, 0.35258494, -0.06174621]


def scale_vector(vec):
    """
    Scale all the entries in the provided vector by the average of the entries.
    :param vec: Vector represented as numpy array of floats
    :return: The scaled vector
    """

    mean = np.mean(vec)
    scaled = vec * mean

    return scaled


scale_vector(np.array(VEC)).tolist()
