"""
You have been given a collection of stock information for Facebook from 2012 (when it went public)
to 2017. Each row of the data contains information for the stock on a day: the date, the opening share price,
the highest share price of the day, the lowest share price of the day, the closing price, and
the volume of the stock.

Calculate the month and year where the stock achieved its lowest closing price.

The data is derived from a collection of all US stocks released
with a CC0 public domain license, available in its original form here: https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs
"""

import pandas as pd
from datetime import datetime

DATA_FILE = "fb_stock.csv"


def get_lowest_close_month(data_file):
    """
    Get month where the stock achieved its lowest closing price. Return the month
    as a string of the format: Year-Month (i.e. August 2020 would be 2020-8)
    :param data_file: Provided csv with data
    :return: The returned month formatted as described above
    """
    df = pd.read_csv(data_file)

    df.Date = pd.to_datetime(df.Date)

    # Resample to months and do minimum aggregation
    df = df.resample("M", on="Date").min()

    # Find min month and convert to Date object
    min_month = datetime.date(df[df.Close == df.Close.min()].index[0])

    minim = "{year}-{month}".format(year=min_month.year, month=min_month.month)

    return minim


get_lowest_close_month(DATA_FILE)