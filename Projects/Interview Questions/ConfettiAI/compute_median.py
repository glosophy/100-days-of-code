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


def compute_median(prices):
    """
    TODO: Fill in this function! Don't use any external libraries :)
    :param prices: A list of car prices.
    :return: Computed price provided as float.
    """

    global median
    prices = prices.sort()

    half = len(prices) / 2

    if isinstance(half, float):
        new_half = len(prices) // 2
        median = (prices[new_half] + prices[new_half + 1]) / 2

    if isinstance(half, int):
        median = prices[half]

    return median


compute_median(ALL_PRICES)
