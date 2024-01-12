# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:31:51 2023

@author: geolki
"""

import pandas as pd
from datetime import datetime
import psutil
import inspect

stats = pd.DataFrame()
start_time_all = datetime.now()

# add timestamp
def add_time_mem(stats, start_time_all, year):
    temp = pd.DataFrame(psutil.virtual_memory()).T
    temp.columns=['total', 'available', 'percent', 'used', 'free']
    temp['duration'] = datetime.now() - start_time_all
    temp['line_no'] = int(inspect.getframeinfo(inspect.stack()[1][0]).lineno)
    temp['year'] = year
    stats = stats.append(temp)
    print(stats)
    return(stats)

stats = add_time_mem(stats, start_time_all, 0)
