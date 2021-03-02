"""
Imagine that you are building a data-driven car price prediction service. As part of
this service you need to pull all the available car prices from an external database.
Provide an implementation of a function for computing the mean car price from the values
in the database.
"""

ALL_PRICES = [
    9961,
    6219,
    6848,
    9007,
    4935,
    4825,
    2357,
    2604,
    1216,
    2796,
    9726,
    6759,
    4620,
    3609,
    1198
]

def compute_mean(prices):
    """
    TODO: Fill in this function! Don't use any external libraries :)
    :param prices: A list of car prices.
    :return: Computed price provided as float.
    """
    mean = sum(prices) / len(prices)
    return mean


compute_mean(ALL_PRICES)
