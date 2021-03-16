"""
Given a vector of float, write a function that computes the nth percentile of the vector for any `n`.
It is common to have n=50, 90, and 99 when analyzing latency data, often referred to as the p50, p90, and p99.
"""
import numpy as np

VEC = [0.6583596987271446, 1.0128241391924433, 0.37783705753739877, 0.42421340135829255, -0.6905233695318467,
       -0.498554227530507]


def compute_percentile(vec, n):
    """
    Computes the nth percentile of the provided data.
    :param vec: Vector of floats provided as numpy array
    :param n: The percentile to compute
    :return: The percentile value
    """
    vec.sort()

    index = (len(vec) - 1) * n / 100

    min_value = int(index // 1)  # floor
    max_value = int(-(-index // 1))  # ceil

    if max_value == min_value:
        return vec[int(index)]

    d0 = vec[int(min_value)] * (max_value - index)
    d1 = vec[int(max_value)] * (index - min_value)

    return d0 + d1


compute_percentile(np.array(VEC), 50)
compute_percentile(np.array(VEC), 90)
compute_percentile(np.array(VEC), 99)
