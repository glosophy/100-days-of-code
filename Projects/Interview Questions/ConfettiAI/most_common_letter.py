"""
You have been given a collection of baby names in the US along with their frequency
from 2011-2013. Determine the most frequent letter used in names for each year.

The data is derived from a collection of US baby names released
with a CC0 public domain license, available in its original form here: https://www.kaggle.com/kaggle/us-baby-names
"""

import pandas as pd

DATA_FILE = "2011_to_2013_names.csv"


def get_most_frequent_letter(data_file):
    """
    Get the most frequent letter used in names for each year.
    :param data_file: Provided csv with data
    :return: The most frequent letter used in names each year as a list of lowercased strings
    """

    df = pd.read_csv(data_file)

    df["first_letter"] = df.Name.apply(lambda name: name.lower()[0])
    letter_counts = df.groupby("Year")["first_letter"].value_counts()
    years = letter_counts.index.get_level_values(0).unique().tolist()
    most_common = [letter_counts.loc[year].head(1).index[0] for year in years]

    return most_common


get_most_frequent_letter(DATA_FILE)
