"""
You have been provided with an HTML dump of a Wikipedia page on the tallest buildings in the
world and you are asked to perform some analysis on the tabular information in the dump.

Implement functionality to compute the average height (in feet) of the tallest buildings, using
the pinnacle height rather than the architectural one.

HINT: Look at the "Alternative Measurements" table. If needed, you can find the original page here:
https://en.wikipedia.org/wiki/List_of_tallest_buildings
"""

from bs4 import BeautifulSoup
import numpy as np

DATA_FILE = "tallest_buildings.html"


def extract_average_height(data_file):
    """
    Compute the average height of the tallest buildings by pinnacle height.
    Try to do this without using the Pandas `read_html` function.
    :param data_file: Provided path to the HTML dump
    :return: The computed average height of the buildings
    """

    with open(data_file) as f:
        soup = BeautifulSoup(f.read(), "html5lib")
    tables = soup.select("table.wikitable.sortable")
    rows = tables[1].select("tr")

    heights = []

    for row in rows[1:]:
        height = float(row.select("td")[-3].contents[0].replace("ft", "").replace(",", "").strip())
        heights.append(height)

    avg_height = np.mean(heights)

    return avg_height


extract_average_height(DATA_FILE)