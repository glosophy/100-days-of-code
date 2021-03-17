"""
You have been tasked with analyzing a baseball dataset, containing information about
all-star appearances from 1933-1942. You are provided a .csv file with the relevant
data. The provided data is structured so that the first row contains the names for
the data columns.

Implement functionality to compute the top 3 teams by number of all-star appearances in 1936.

Data adapted from http://www.seanlahman.com/baseball-archive/statistics/ under CC BY-SA 3.0
"""

import pandas as pd

DATA_FILE = "baseball_data.csv"
YEAR = 1936


def top_all_star_appearances(data_file):
    """
    Compute top 3 teams by number of all-star appearances.
    :param data_file: Provided csv with data
    :return: A list of the team IDs for the top 3 teams.
    """

    df = pd.read_csv(data_file)
    team_count_players = df[df.yearID == YEAR].groupby("teamID")["playerID"].count()
    top3 = team_count_players.sort_values(ascending=False)[:3].index

    return top3


top_all_star_appearances(DATA_FILE)