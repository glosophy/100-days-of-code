"""
You have been given a collection of baby names in the US along with their frequency
from 2011-2013. Determine the most frequent male name given each year.

The data is derived from a collection of US baby names released
with a CC0 public domain license, available in its original form here: https://www.kaggle.com/kaggle/us-baby-names
"""

import pandas as pd

DATA_FILE = "2011_to_2013_names.csv"


def get_most_frequent_male_names(data_file):
    """
    Get the most frequent male name given each year, with one name for each year.
    :param data_file: Provided csv with data
    :return: A list containing the most frequent male name each year (chronological order) for all
             years in the dataset
    """

    df = pd.read_csv(data_file)

    df = df.sort_values('Count', ascending=False).groupby(["Year", "Gender"]).first()
    df = df.reset_index(level='Gender')
    male_names = df[df.Gender == "M"].Name.tolist()

    return male_names


get_most_frequent_male_names(DATA_FILE)
