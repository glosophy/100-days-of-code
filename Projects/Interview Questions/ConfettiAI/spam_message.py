"""
You have been given a collection of text messages some of which are spam and
some of which are not-spam (i.e. ham).

Get the average number of words (tokens) in all the spam messages.

The data is derived from a collection of SMS messages released publicly, available in its original form here:
https://www.kaggle.com/uciml/sms-spam-collection-dataset
http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/.

Citation:
Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. Contributions to the Study of SMS Spam Filtering: New Collection and Results.  Proceedings of the 2011 ACM Symposium on Document Engineering (DOCENG'11), Mountain View, CA, USA, 2011.
"""

import pandas as pd

DATA_FILE = "spam.csv"


def get_spam_message_length(data_file):
    """
    Get the average number of words across all the spam messages. Here split words
    of the message using whitespace.
    :param data_file: Provided csv with data
    :return: The average number of words in spam messages
    """
    df = pd.read_csv(data_file)

    spam = df[df['label'] == 'spam']

    average_len = spam['message'].str.split().apply(len).mean()

    return average_len


get_spam_message_length(DATA_FILE)