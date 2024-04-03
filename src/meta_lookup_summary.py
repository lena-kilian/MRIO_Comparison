# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 15:57:01 2024

@author: geolki
"""

import pandas as pd
from sys import platform
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import copy as cp

# set working directory
# make different path depending on operating system
if platform[:3] == 'win':
    wd = 'O://'
else:
    wd = r'/Volumes/a72/' 

# define filepaths
data_filepath = wd + 'ESCoE_Project/data/'
emissions_filepath = wd + 'ESCoE_Project/data/Emissions/'
plot_filepath = wd + 'ESCoE_Project/outputs/compare_all_outputs/plots/'

data_dict = {'oecd':'ICIO', 'exio':'Exiobase', 'gloria':'Gloria', 'figaro':'Figaro'}

meta = pd.read_excel(data_filepath + 'lookups/mrio_lookup_paper.xlsx', sheet_name=None)

# countries

countries = meta['countries'][['combined_name', 'exio', 'figaro', 'gloria', 'oecd']].apply(lambda x: x + '; ')
countries['All'] = countries['combined_name']
countries.loc[countries['All'] != 'Rest of the World; ', 'All'] = 'Same'
countries = pd.DataFrame(countries.set_index(['combined_name', 'All']).stack())
countries['count'] = 1                      
countries = countries.sum(axis=0, level=[1, 2])
countries['list'] = countries[0].str.split(', ')
countries['list'] = [sorted(x) for x in countries['list']]
countries['list'] = [''.join(x) for x in countries['list']]
countries['list'] = [x[:-2] for x in countries['list']]
countries = countries[['list', 'count']].rename(index=data_dict)

# sectors

temp = meta['sectors'][['combined_name', 'combined_code', 'exio', 'figaro', 'gloria', 'oecd']].apply(lambda x: x + '; ')

sectors = pd.DataFrame(index = temp['combined_name'].unique())
for item in list(data_dict.keys()):
    temp2 = temp[['combined_name', item]].drop_duplicates().dropna(how='any').sort_values(item)
    temp3 = cp.copy(temp2); temp3['count'] = 1; temp3 = temp3.groupby('combined_name').sum()
    temp2 = temp2.groupby('combined_name').sum()
    temp2 = temp2.join(temp3)
    temp2[item] = [x[:-2] for x in temp2[item]]
    sectors[data_dict[item]] = temp2[item]
    sectors['count_' + item] = temp2['count']
sectors = sectors.join(temp.set_index('combined_name')[['combined_code']].drop_duplicates(), how='left')
sectors.index = [x[:-2] for x in sectors.index]

# final demand

temp = meta['final_demand'][['combined_name', 'exio', 'figaro', 'gloria', 'oecd']].apply(lambda x: x + '; ')

final_demand = pd.DataFrame(index = temp['combined_name'].unique())
for item in list(data_dict.keys()):
    temp2 = temp[['combined_name', item]].drop_duplicates().dropna(how='any').sort_values(item)
    temp3 = cp.copy(temp2); temp3['count'] = 1; temp3 = temp3.groupby('combined_name').sum()
    temp2 = temp2.groupby('combined_name').sum()
    temp2 = temp2.join(temp3)
    temp2[item] = [x[:-2] for x in temp2[item]]
    final_demand[data_dict[item]] = temp2[item]
    final_demand['count_' + item] = temp2['count']
final_demand.index = [x[:-2] for x in final_demand.index]