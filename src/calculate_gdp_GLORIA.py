# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:31:51 2023

@author: geolki
"""

import pandas as pd
from sys import platform
import import_function_gloria_gdp as imp_g
import numpy as np

# set working directory
# make different path depending on operating system
if platform[:3] == 'win':
    wd = 'O://'
else:
    wd = r'/Volumes/a72/' 

# define filepaths
mrio_filepath = wd + 'ESCoE_Project/data/MRIO/'
emissions_filepath = wd + 'ESCoE_Project/data/Emissions/'

version = '2024'
footprint = 'ghg'
years = range(2005, 2006)

############
## Gloria ##
############

lookup_cat =  'abc' #'combined_name' #  # 
row_lookup = pd.read_excel('O:/ESCoE_Project/data/lookups/GLORIA_small_sectors.xlsx', sheet_name='Countries')[['gloria_code', lookup_cat]].drop_duplicates()
row_dict = dict(zip(row_lookup['gloria_code'], row_lookup[lookup_cat]))

# read config file to get filenames
config_file= wd + 'ESCoE_Project/data/MRIO/Gloria/config_large.cfg'
gloria_filepath, outdir, lookup_filepath, labels_fname, lookup_fname, Z_fname, Y_fname, co2_fname, gloria_version = imp_g.read_config(config_file)

z_idx, industry_idx, product_idx, iix,pix, y_cols, v_cols, sat_rows = imp_g.get_metadata_indices(gloria_filepath, lookup_filepath, labels_fname, lookup_fname)


# define sample year, normally this is: range(2010, 2019)
# here years is now determined from inputs,
# it used to be a range(2010, 2019). In future work this will likley be range(2001, 2023)
V = {}
for year in years:
    print('start', year)

    split=Y_fname.split('%')
    if len(split)>1:
        y_filepath=gloria_filepath+gloria_version+split[0]+str(year)+split[1]
    else:
        y_filepath=gloria_filepath+gloria_version+Y_fname
        
    v_filepath = y_filepath.replace('_Y_', '_V_')
        
    V[year] = imp_g.read_data_new(v_filepath, iix, pix, industry_idx, product_idx, v_cols)
   
    print('end', year)