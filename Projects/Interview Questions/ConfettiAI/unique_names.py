"""
You have been given a collection of baby names in the US along with their frequency how many babies
had that name in each year from 2011-2013. Count how many unique names there were in each year provided
in the dataset.

The data is derived from a collection of US baby names released
with a CC0 public domain license, available in its original form here: https://www.kaggle.com/kaggle/us-baby-names
"""

import pandas as pd

DATA_FILE = "2011_to_2013_names.csv"


def get_num_unique_names(data_file):
    """
    Count the number of unique names in each year, ordered chronologically by the year.
    :param data_file: Provided csv with data
    :return: A list containing the number of unique names per year sorted chronologically
    """

    df = pd.read_csv(data_file)

    names = df.groupby("Year").Name.nunique().tolist()

    return names


get_num_unique_names(DATA_FILE)
