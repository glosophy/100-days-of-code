"""
You have been given a collection of text messages some of which are spam and
some of which are not-spam (i.e. ham).

Find the number of tokens across all messages that contain a digit. For example
if you have the message "hi 3 there a5a", you'll return 2.

The data is derived from a collection of SMS messages released publicly, available in its original form here:
https://www.kaggle.com/uciml/sms-spam-collection-dataset
http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/.

Citation:
Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. Contributions to the Study of SMS Spam Filtering: New Collection and
Results.  Proceedings of the 2011 ACM Symposium on Document Engineering (DOCENG'11), Mountain View, CA, USA, 2011.
"""

import pandas as pd

DATA_FILE = "spam.csv"

def get_digit_tokens(data_file):
    """
    Get the number of tokens across all the messages that contain a digit. Here tokens
    of the message come from simple splitting on whitespace.
    :param data_file: Provided csv with data
    :return: The total number of tokens containing a digit
    """
    df = pd.read_csv(data_file)

    tokens = df.message.str.split(expand=True).stack()

    total_tokens = sum(tokens.str.contains("[\d]+"))

    return total_tokens


get_digit_tokens(DATA_FILE)
