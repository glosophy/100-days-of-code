"""
Given a `start` and `end` integer values, create a 1-d array consisting of all
integer values between `start` and `end` (inclusive).
"""

import numpy as np


def create_range(start, end):
    """
    Create a range of values between `start` and `end` inclusive.
    :param start: Lower bound value
    :param end: Upper bound value
    :return: A 1-d array with the range of integer values.
    """
    l = np.arange(start, end + 1, 1)

    return l


# NOTE: Do not modify the below lines!
create_range(3, 10).tolist()