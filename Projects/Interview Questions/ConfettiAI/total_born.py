"""
You have been given a collection of baby names in the US along with their frequency
from 2011-2013. Determine how many babies were born in 2013.

The data is derived from a collection of US baby names released
with a CC0 public domain license, available in its original form here: https://www.kaggle.com/kaggle/us-baby-names
"""

import pandas as pd

DATA_FILE = "2011_to_2013_names.csv"
YEAR = 2013


def get_num_babies_born(data_file, year):
    """
    Get the total number of babies born in the provided year.
    :param data_file: Provided csv with data
    :param year: Year to analyze, given as an int
    :return: The number of babies born in a certain year
    """

    df = pd.read_csv(data_file)
    total_born = df[df['Year'] == year].Count.sum()

    return total_born


get_num_babies_born(DATA_FILE, YEAR)