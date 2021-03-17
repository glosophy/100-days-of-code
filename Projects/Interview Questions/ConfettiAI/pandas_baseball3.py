"""
You have been tasked with analyzing a baseball dataset, containing information about
all star appearances from 1933-1942. You are provided a .csv file with the relevant
data. The provided data is structured so that the first row contains the names for
the data columns.

Implement functionality to compute the median and standard deviation of the number of all stars per year.

Data adapted from http://www.seanlahman.com/baseball-archive/statistics/ under CC BY-SA 3.0
"""

import pandas as pd

DATA_FILE = "baseball_data.csv"


def all_star_summary_statistics(data_file):
    """
    Compute the median and standard deviation of the number of all star players per year.
    :param data_file: Provided csv with data
    :return: A tuple containing the median and stddev (median, stddev)
    """

    df = pd.read_csv(DATA_FILE)
    players = df.groupby("yearID")["playerID"].count()
    median = players.median()
    std = players.std()

    return median, std


list(all_star_summary_statistics(DATA_FILE))
