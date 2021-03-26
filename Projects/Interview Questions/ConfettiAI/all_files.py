"""
You have been given multiple files with a students with their ages at a small
kindergarden through 12th grade school.

Find the average age of all the students.
"""

import pandas as pd

DATA_FILE1 = "students1.csv"
DATA_FILE2 = "students2.csv"


def get_average_age_of_students(*data_files):
    """
    Get the average age of all the students contained in the files provided.
    :param data_files: A variable-length list of csv files
    :return: The average age across all students
    """

    all_df = [pd.read_csv(data_file) for data_file in data_files]
    merged_df = pd.concat(all_df, axis=0, ignore_index=True)

    age_mean = merged_df.age.mean()

    return age_mean


get_average_age_of_students(DATA_FILE1, DATA_FILE2)