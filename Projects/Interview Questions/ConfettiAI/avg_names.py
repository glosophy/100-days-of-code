"""
You have been given a collection of baby names in the US along with their frequency
from 2011-2013. Determine the average length of names (in characters) for all names given in 2012.

The data is derived from a collection of US baby names released
with a CC0 public domain license, available in its original form here: https://www.kaggle.com/kaggle/us-baby-names
"""

import pandas as pd

DATA_FILE = "2011_to_2013_names.csv"
YEAR = 2012


def get_avg_name_length(data_file, year):
    """
    Get the average length of names (in characters) in the given year.
    :param data_file: Provided csv with data
    :param year: Year to analyze, given as an int
    :return: The average length of names in the given year as a float
    """

    df = pd.read_csv(data_file)
    avg_length = df[df.Year == year].Name.str.len().mean()

    return avg_length


get_avg_name_length(DATA_FILE, YEAR)