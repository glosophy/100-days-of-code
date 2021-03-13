"""
Given  vector of data, create a new vector where every value greater than 1 is truncated to 1.
"""

import numpy as np

VEC = [-0.020, 1.49908079, 1.45047058, 1.3897799, -0.5208,
       -0.061, 1.94843238, 0.75593, -0.24214, -0.8718]


def conditional_truncation(vec):
    """
    Truncate the vector based on values greater than 1.
    :param vec: Vector represented as a numpy array
    :return: The transformed vector
    """
    vec = np.where(vec > 1, 1, vec)

    return vec


conditional_truncation(np.array(VEC)).tolist()