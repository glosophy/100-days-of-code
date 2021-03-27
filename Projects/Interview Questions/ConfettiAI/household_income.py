import os
import tarfile
import numpy as np
from pathlib import Path
import pandas as pd

cwd = os.getcwd()

extracted_to_path = Path.cwd()
with tarfile.open('marketing1.tar.gz') as tar:
    tar.extractall(path=cwd)

# read txt file with pandas

