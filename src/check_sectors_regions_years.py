# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:31:51 2023

@author: geolki
"""

import pandas as pd
from sys import platform
import numpy as np


'''
Figaro is available from 2010
ICIO is available until 2018

So year range should be 2010-2018
'''

# set working directory
# make different path depending on operating system
if platform[:3] == 'win':
    wd = 'O://'
else:
    wd = r'/Volumes/a72/' 

# define filepaths
data_filepath = wd + 'UKMRIO_Data/data/'
outputs_filepath = wd + 'UKMRIO_Data/outputs/results_2023/'

# use 2018 as example to get meta data
year = 2018

#################
## Import data ##
#################

exio = pd.read_csv(wd + 'UKMRIO_Data/EXIOBASE/IOT/IOT_' + str(year) + '_ixi/Y.txt', sep='\t', header = [0, 1], index_col = [0, 1])

figaro = pd.read_excel(wd + 'UKMRIO_Data/Figaro/Description-FIGARO-tables.xlsx', sheet_name=None)

oecd = pd.read_excel(wd + 'UKMRIO_Data/ICIO/ReadMe_ICIO2021_CSV.xlsx', sheet_name=None)

gloria = pd.read_excel(wd + 'UKMRIO_Data/Gloria/GLORIA_ReadMe_057.xlsx', sheet_name=None)


# Countries
countries = {}

temp = pd.read_excel(wd + 'UKMRIO_Data/EXIOBASE/IOT/CountryMappingDESIRE.xlsx', sheet_name='CountryList')[
    ['ISO2', 'Name', 'DESIRE region', 'DESIRE region name']]
lookup = temp.iloc[:,:2]; lookup.columns = ['code', 'name']
temp = temp.iloc[:,2:]; temp.columns = ['code', 'name']
lookup = pd.concat([lookup, temp]).drop_duplicates().dropna(how='any')
countries['exio'] = pd.DataFrame(exio.index.levels[0].tolist()).set_index(0)
countries['exio'] = countries['exio'].join(lookup.set_index('code'), how='left').reset_index().set_index('name').drop_duplicates().reset_index()
countries['exio'].columns = ['exio', 'exio_code']

countries['figaro'] = figaro['Countries'].iloc[10:, 1:]
temp = countries['figaro'].iloc[:,2:].dropna(); temp.columns = ['figaro_code', 'figaro']
countries['figaro'] = countries['figaro'].iloc[:,:2].dropna(); countries['figaro'].columns = ['figaro_code', 'figaro']
countries['figaro'] = pd.concat([countries['figaro'], temp]).dropna(how='any')

countries['oecd'] = oecd['Country_Industry'].iloc[2:, 1:5]
temp = countries['oecd'].iloc[:,2:].dropna(how='any'); temp.columns = ['oecd_code', 'oecd']
countries['oecd'] = countries['oecd'].iloc[:,:2]; countries['oecd'].columns = ['oecd_code', 'oecd']
countries['oecd'] = pd.concat([countries['oecd'], temp]).dropna(how='any')

countries['gloria'] = gloria['Regions'].iloc[:,1:]; countries['gloria'].columns = ['gloria_code', 'gloria']

# merge all into one df
# combine exio and figaro
temp = countries['exio']; temp['match'] = temp['exio_code']
temp2 = countries['figaro']; temp2['match'] = temp2['figaro_code']
countries_temp = temp.merge(temp2, on='match', how='outer')
# combine exio and figaro
temp = countries['oecd']; temp['match'] = temp['oecd_code']
temp2 = countries['gloria']; temp2['match'] = temp2['gloria_code']
temp = temp.merge(temp2, on='match', how='outer')
# merge all
countries_temp['match'] = countries_temp['exio']
countries_temp.loc[countries_temp['exio'].isna() == True, 'match'] = countries_temp['figaro']
temp['match'] = temp['oecd']
countries_temp = countries_temp.merge(temp, on='match', how='outer')

# Sectors
sectors = {}

sectors['exio'] = exio.reset_index()[[('sector', '')]].drop_duplicates(); sectors['exio'].columns = ['exio']

sectors['figaro'] = figaro['Codes'].iloc[9:103, 1:3]; sectors['figaro'].columns = ['figaro_code', 'figaro']

sectors['oecd'] = oecd['Country_Industry'].iloc[2:, 6:8].dropna(how='all'); sectors['oecd'].columns = ['oecd_code', 'oecd']

temp = pd.DataFrame(gloria['GLORIA to ISIC concordance'].set_index('GLORIA ISIC Rev. 4 concordance')\
                    .T.set_index(np.nan, append=True).stack())
sectors['gloria'] = temp.loc[temp[0] != 0].reset_index().drop(0, axis=1)
sectors['gloria'].columns = ['gloria_cat_code', 'gloria_cat', 'gloria']


check=pd.read_excel(data_filepath +'lookups/mrio_lookup_sectors_countries_finaldemand.xlsx', sheet_name='sectors')
for item in ['exio', 'gloria', 'figaro', 'oecd']:
    temp = check[['combined_name', item]].drop_duplicates()
    for var in temp[item]:
        temp2 = temp.loc[temp[item] == var]
        if len(temp2.index)>1:
            print(item, var)

# Final Demand
final_demand = {}

final_demand['exio'] = pd.DataFrame(exio.columns.levels[1].tolist()); final_demand['exio'].columns = ['exio']

final_demand['figaro'] = figaro['Codes'].iloc[204:216, 1:3]; final_demand['figaro'].columns = ['figaro_code', 'figaro']

final_demand['oecd'] = oecd['Structure'].iloc[38:,].dropna(how='all', axis=1); final_demand['oecd'].columns = ['oecd_code', 'oecd']

final_demand['gloria'] = gloria['Value added and final demand'][['Final_demand_names']]; final_demand['gloria'].columns = ['gloria']

for item in list(final_demand.keys()):
    final_demand[item] = final_demand[item].sort_values(item)