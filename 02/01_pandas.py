import datetime

import pandas as pd
import numpy as np

import time

date = [datetime.datetime(2022,1,1),
        datetime.datetime(2022,1,2),
        datetime.datetime(2022,1,3)]

serie = pd.Series([10,20,30], index=date)
print(serie['2022-01-02'])