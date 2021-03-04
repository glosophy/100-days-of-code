"""
In many machine learning contexts, we wish to find the closest point to some query point according
to some distance measure. This forms the core operation of modeling algorithms such as k-nearest neighbors.

Implement a function to compute the closest point in a collection of points according to a
Euclidean distance. For a reminder of Euclidean distance, check out: https://www.cut-the-knot.org/pythagoras/DistanceFormula.shtml
"""

import numpy as np

POINTS = [[0.4878536768447066,
           0.869456188424337,
           0.5011068612999854,
           1.0133445679243316,
           0.8148628393772428],
          [-1.9546441319243815,
           4.203616681164113,
           -1.2296409861901172,
           -0.7524776363348679,
           0.7636937144300029],
          [-0.8312390019196739,
           -1.7188556451950765,
           1.296113631342589,
           -0.1920154915288227,
           -0.23289363410028704],
          [0.0812036270933463,
           0.3691611770903292,
           -0.25254671963123604,
           -0.698690361395649,
           0.7184178853051904],
          [1.0719775045458273,
           -0.9585189675423648,
           -0.6868941871154016,
           0.2951448085864521,
           -0.40818130509087],
          [0.6189255807478202,
           -1.628248909022074,
           -0.3870923080270297,
           1.0238007396805155,
           -0.43619087044715915],
          [0.6469704980278533,
           -0.5510403453068279,
           0.57974527600909,
           -0.18227673639249417,
           2.431419898529448],
          [-1.2527653438077144,
           -0.04096672475177045,
           0.6979648406646776,
           -0.06805987269966886,
           -0.38933102408171466],
          [-0.8047882379332667,
           0.3906445194669081,
           -0.15367552112839183,
           0.8627565512188639,
           -0.5468546042030066],
          [0.6579256363881655,
           0.08540398264794942,
           -0.3685160479312876,
           -2.2590117617618795,
           -1.0637405525056955]]

QUERY = [0.6784155, 0.21687017, 1.01940089, 3.07288809, -1.16329905]


def compute_closest_point(query, points):
    """
    TODO: Fill in this function!
    :param query: An n-dimensional point represented as a list of floats.
    :param points: A list of n-dimensional points where each point is a list of floats.
    :return: The index of the closest point in the `points` list.
    """

    query = np.array(query)
    points = np.array(points)
    euclidean = np.sqrt(np.square(points - query).sum(axis=1))
    min_distance = np.argmin(euclidean)
    return min_distance


print(compute_closest_point(QUERY, POINTS))
