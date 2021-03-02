"""
Imagine that you are building a service that must process a stream of data
regarding inference latencies of a model. It is common to compute summary statistics
regarding a collection of latency data including its p50 and p90 (the cutoff for values in
the upper 50th and upper 90th percentiles respectively).

Implement functions to compute these values for a collection of data.
"""

ALL_LATENCIES = [
    0.25589525815118686,
    0.1666865011880021,
    0.3640850051711415,
    0.46888554561214413,
    1.5264772687981958,
    1.2182882133438868,
    0.2963250423370145,
    2.070337738926535,
    5.32887226557879,
    3.5611535580285754,
    0.6357744879510403,
    2.609330343370753,
    3.167112136066649,
    1.9448104942989919,
    0.3110763092201544,
    2.039493671452421,
    0.43159245938645563,
    2.500865500659632,
    0.8442174772740363,
    1.998084024508116
]


def calculate_percentile(latencies, percentile):
    latencies.sort()

    index = (len(latencies) - 1) * percentile / 100

    min_value = int(index // 1)  # floor
    max_value = int(-(-index // 1))  # ceil

    if max_value == min_value:
        return latencies[int(index)]

    d0 = latencies[int(min_value)] * (max_value - index)
    d1 = latencies[int(max_value)] * (index - min_value)
    return d0 + d1


def compute_p50(latencies):
    """
    TODO: Fill in this function! Don't use any external libraries :)
    :param latencies: A list of latencies.
    :return: Computed p50 provided as float.
    """
    return calculate_percentile(latencies, 50)


def compute_p90(latencies):
    """
    TODO: Fill in this function! Don't use any external libraries :)
    :param latencies: A list of latencies.
    :return: Computed p90 provided as float.
    """
    return calculate_percentile(latencies, 90)


compute_p50(ALL_LATENCIES)
compute_p90(ALL_LATENCIES)
