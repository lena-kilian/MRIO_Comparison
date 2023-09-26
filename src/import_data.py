# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:31:51 2023

@author: geolki
"""

# use pymrio https://pymrio.readthedocs.io/en/latest/notebooks/autodownload.html

import pymrio
import os
import pandas as pd
import numpy as np
df = pd.DataFrame
from sys import platform


# set working directory
# make different path depending on operating system
if platform[:3] == 'win':
    wd = 'O://'
else:
    wd = r'/Volumes/a72/' 

# define filepaths
inputs_filepath = wd + 'UKMRIO_Data/data/model_inputs/'
mrio_data_path = wd + 'geolki/data/raw/MRIOs/'

db = ['Exiobase3', 'OECD', 'Figaro', 'Gloria', 'WIOD', 'EORA']


# define filepaths

##############
## EXIOBASE ##
##############

exioyrs = range(1995, 2021)

for year in exioyrs:
                  
    filepath = exiobase_filepath + "3.8.2/MRSUT_{}/".format(str(year))
            
    exio_s = pd.read_csv(filepath + 'supply.csv', sep='\t', header = [0,1], index_col = [0,1])
    exio_u = pd.read_csv(filepath + 'use.csv', sep='\t', header = [0,1], index_col = [0,1])
    exio_y = pd.read_csv(filepath + 'final_demand.csv', sep='\t', header = [0,1], index_col = [0,1])
    exio_v = pd.read_csv(filepath + 'value_added.csv', sep='\t', header = [0,1], index_col = 0)
    exio_v = df.sum(exio_v.iloc[0:12,:], 0)