"""
Given a set of vectors of floats, find the common positive entries among them. Return your results
as a sorted list of values.
"""
import numpy as np
from functools import reduce

VEC_1 = [-0.25560104, 0.06393334, -0.43760861, 0.35258494, -0.06174621, -1.11846337, -0.83336927]
VEC_2 = [-0.23099469, -0.25560104, -0.45461739, 0.35258494, -1.11846337]
VEC_3 = [0.35258494, -1.97636123, 0.77339655, -0.25560104, -1.11846337]


def compute_common_entries(*vecs):
    """
    Computes the common unique, *positive* entries among the provided `vecs`, returned in ascending order.
    :param vecs: Variable-length args with vectors of data to find common elements among
    :return: The common positive entries among the `vecs` sorted in ascending order.
    """

    common = reduce(np.intersect1d, (vecs))

    pos = common[common > 0]

    return pos


compute_common_entries(np.array(VEC_1), np.array(VEC_2), np.array(VEC_3)).tolist()
