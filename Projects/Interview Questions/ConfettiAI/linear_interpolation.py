"""
Given the input data of points [(1, 2), (3,4), (5,6), (8, 8), (7, -1)], fit a line of the form
Y = A * X + B using gradient descent. Provide the implementation of your algorithm in the function provided.
Then evaluate your linear regression fit on the provided point.
"""
import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_iters):
    n_samples = len(y)
    # NOTE: A better strategy is to compute the cost at each step and run until
    # the cost converges to a small theshold value
    for i in range(n_iters):
        weights = weights - learning_rate * np.dot(X.T, np.dot(X, weights) - y.reshape(-1, 1))

    return weights


def linear_interpolate_point(data, test_points):
    """
    Fit a line to the provided data and then evaluate on the provided test point.
    :param data: Collection of points to fit provided as a list of tuples
    :param x: Point to interpolate using your fit line
    :return: The output of your point on the interpolated line
    """
    # fill in the function here
    learning_rate = 0.01
    n_iters = 2000
    weights = np.zeros(2).reshape(-1, 1)
    data = np.array(data)
    X = np.ones_like(data)
    X[:, 0] = data[:, 0]
    weights = gradient_descent(X, data[:, 1], weights, learning_rate, n_iters)
    return test_points * weights[0, 0] + weights[1, 0]


linear_interpolate_point(
    [(1, 2), (3,4), (5,6), (8, 8), (7, -1)],
    10.5)