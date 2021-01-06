import urllib
import pandas as pd

data_urls = ["https://worldjusticeproject.org/sites/default/files/documents/FINAL_2020_wjp_rule_of_law_index_HISTORICAL_DATA_FILE_1.xlsx",
             'https://drive.google.com/uc?export=download&id=0BxDpF6GQ-6fbTUJZOExlM0FQVFk',
             'https://ucdp.uu.se/downloads/brd/ucdp-brd-conf-201-xlsx.zip',
             ]

url_names = ['ROL', 'CIRI', 'Uppsala', ]

for i in range(len(data_urls)):
    urllib.request.urlretrieve(data_urls[i], "{}.xls".format(url_names[i]))

