# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:31:51 2023

@author: geolki
"""

import pandas as pd
from sys import platform

# set working directory
# make different path depending on operating system
if platform[:3] == 'win':
    wd = 'O://'
else:
    wd = r'/Volumes/a72/' 

# define filepaths
input_filepath = 'C:/Users/geolki/OneDrive - University of Leeds/Postdoc/'

footprint = 'ghg'
years = range(2005, 2020)
    
############
## Gloria ##
############

data = {}; fd = {}
for year in years:

    emissions_ind = pd.read_csv(input_filepath + 'Riepl_paper/Gloria_all/Gloria_' + footprint + '_industries_' + str(year) + '_governments.csv', header=[0, 1], index_col=[0, 1])
    emissions_prod = pd.read_csv(input_filepath + 'Riepl_paper/Gloria_all/Gloria_' + footprint + '_products_' + str(year) + '_governments.csv', header=[0, 1], index_col=[0, 1])

    emissions_ind = pd.DataFrame(emissions_ind.stack(level=[0, 1]), columns=['industries'])
    emissions_ind.index.names = ['origin_country', 'origin_sector', 'fd_country', 'fd_cat']
    emissions_ind = emissions_ind.reset_index()
    emissions_ind['origin_sector'] = emissions_ind['origin_sector'].str.replace(' industry', '')
    
    emissions_prod = pd.DataFrame(emissions_prod.stack(level=[0, 1]), columns=['products'])
    emissions_prod.index.names = ['origin_country', 'origin_sector', 'fd_country', 'fd_cat']
    emissions_prod = emissions_prod.reset_index()
    emissions_prod['origin_sector'] = emissions_prod['origin_sector'].str.replace(' industry', '')
    
    data[year] = emissions_ind.merge(emissions_prod, on=['origin_country', 'origin_sector', 'fd_country', 'fd_cat'])
    
    print(year, data[year].corr().loc['industries', 'products'])
    
    'C:/Users/geolki/OneDrive - University of Leeds/Postdoc/Gloria_detail/Gloria_spend/FD_Gloria_' + footprint + '_products_' + str(year) + '_all.csv'