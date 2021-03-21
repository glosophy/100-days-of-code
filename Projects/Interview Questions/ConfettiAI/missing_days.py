"""
You have been given a collection of stock information for Facebook from 2012 (when it went public)
to 2017. Each row of the data contains information for the stock on a day: the date, the opening share price,
the highest share price of the day, the lowest share price of the day, the closing price, and
the volume of the stock.

Find how many days the market was closed (in other words, how many days are missing in the dataset).

The data is derived from a collection of all US stocks released
with a CC0 public domain license, available in its original form here: https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs
"""

import pandas as pd

DATA_FILE = "fb_stock.csv"


def get_num_missing_days(data_file):
    """
    Get the number of days we don't have information for within the dataset.
    :param data_file: Provided csv with data
    :return: The total number of missing days as an int
    """
    df = pd.read_csv(data_file)

    df.Date = pd.to_datetime(df.Date)

    start_interval = df.iloc[0].Date
    end_interval = df.iloc[len(df) - 1].Date

    # Gets all the days in the provided time range
    stock_range = pd.date_range(start_interval, end_interval)

    missing_dates = stock_range[~stock_range.isin(df.Date)]

    total_missing_days = len(missing_dates)

    return total_missing_days


get_num_missing_days(DATA_FILE)