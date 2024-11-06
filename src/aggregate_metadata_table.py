# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 10:30:46 2023

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

# define version
version = '2024'
footprint = 'ghg'

# define filepaths
data_filepath = wd + 'ESCoE_Project/data/'
emissions_filepath = wd + 'ESCoE_Project/data/Emissions/'

# Load lookup file
lookup = pd.read_excel(data_filepath + 'lookups/mrio_lookup_sectors_countries_finaldemand.xlsx', sheet_name = None)

data_dict = {'oecd':'ICIO', 'exio':'Exiobase', 'gloria':'Gloria', 'figaro':'Figaro'}

# countries
country_dict = {'United Kingdom':'UK', 'Czech Republic':'Czechia', 'United States':'USA', 'Rest of the World':'RoW'}

countries = lookup['countries'].set_index('combined_name')[['exio', 'figaro', 'oecd', 'gloria']].stack()\
    .rename(index=country_dict).reset_index().drop_duplicates().sort_values([0, 'combined_name'])
countries.loc[countries['combined_name'] != 'RoW', 0] = countries['combined_name']
countries.loc[countries['combined_name'] != 'RoW', 'combined_name'] = 'Countries'

list_c = pd.DataFrame(countries['combined_name'].unique())

countries[0] = countries[0] + ', '
countries = countries.groupby(['combined_name', 'level_1']).sum()
countries[0] = countries[0].str[:-2]
countries = countries.unstack().droplevel(axis=1, level=0).rename(columns=data_dict)
