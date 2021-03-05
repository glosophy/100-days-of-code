"""
Given the input data of points [(1, 2), (3,4), (5,6), (8, 8), (7, -1)], fit a line of the form
Y = A * X + B using gradient descent. Provide the implementation of your algorithm in the function provided.
Then evaluate your linear regression fit on the provided point.
"""
import numpy as np

def linear_interpolate_point(data, test_points):
    """
    Fit a line to the provided data and then evaluate on the provided test point.
    :param data: Collection of points to fit provided as a list of tuples
    :param x: Point to interpolate using your fit line
    :return: The output of your point on the interpolated line
    """
    # fill in the function here
    pass

linear_interpolate_point(
    [(1, 2), (3,4), (5,6), (8, 8), (7, -1)],
    10.5
)