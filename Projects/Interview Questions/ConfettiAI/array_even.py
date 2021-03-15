"""
Given a vector of integer data, write a function to return all the entries that are even.
"""
import numpy as np

VEC = [-3, 5, 1, 2, 18, 2, 234, 11]


def compute_even(vec):
    """
    Computes the even integers in `vec1`.
    :param vec1: Vector represented as numpy array of integers
    :return: The even entries in the vector as a numpy array
    """

    even = []

    for i in range(len(vec)):
        if vec[i] % 2 == 0:
            even.append(vec[i])

    return even


# NOTE: Do not modify the below lines!
compute_even(np.array(VEC))
