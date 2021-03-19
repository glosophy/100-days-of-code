"""
You have been given a collection of stock information for Facebook from 2012 (when it went public)
to 2017. Each row of the data contains information for the stock on a day: the date, the opening share price,
the highest share price of the day, the lowest share price of the day, the closing price, and
the volume of the stock.

Calculate how many months of stock information we have in this dataset.

The data is derived from a collection of all US stocks released
with a CC0 public domain license, available in its original form here: https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs
"""

import pandas as pd

DATA_FILE = "fb_stock.csv"


def get_number_of_months(data_file):
    """
    Get the number of months of stock data in the dataset.
    :param data_file: Provided csv with data
    :return: The number of months returned as an int
    """
    df = pd.read_csv(data_file)

    df.Date = pd.to_datetime(df.Date)

    # Resample to months and do minimum aggregation
    df = df.resample("M", on="Date")

    number_months = len(df)

    return number_months


get_number_of_months(DATA_FILE)