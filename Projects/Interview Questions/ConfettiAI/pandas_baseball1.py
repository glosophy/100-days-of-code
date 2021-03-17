"""
You have been tasked with analyzing a baseball dataset, containing information about
all-star appearances from 1933-1942. You are provided a .csv file with the relevant
data. The provided data is structured so that the first row contains the names for
the data columns.

Implement functionality to compute how many distinct all-stars there were in 1934.

Data adapted from http://www.seanlahman.com/baseball-archive/statistics/ under CC BY-SA 3.0
"""

import pandas as pd

DATA_FILE = "baseball_data.csv"
YEAR = 1934


def count_all_stars(data_file):
    """
    Count all the unique all-stars in the provided file.
    :param data_file: Provided csv with data
    :return: An integer represnting the number of distinct all-stars
    """
    df = pd.read_csv(data_file)
    df_filter = df[df.yearID == YEAR].playerID.unique()
    all_stars = len(df_filter)

    return all_stars


count_all_stars(DATA_FILE)
