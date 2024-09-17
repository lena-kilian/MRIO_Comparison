# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:31:51 2023

@author: geolki
"""

# conda activate mrio

import pandas as pd
from sys import platform
import calculate_emissions_functions as cef
import pickle
import numpy as np
import copy as cp


# set working directory
# make different path depending on operating system
if platform[:3] == 'win':
    wd = 'O://'
else:
    wd = r'/Volumes/a72/' 

# define filepaths
mrio_filepath = wd + 'ESCoE_Project/data/MRIO/'
emissions_filepath = wd + 'ESCoE_Project/data/Emissions/'

years = range(2010, 2021)

version = '2024'
footprint = 'ghg'

############
## Figaro ##
############

# get figaro footprint
co2_figaro = {}
for year in years:
    co2_figaro[year] = pd.read_csv(mrio_filepath + 'Figaro/Ed_2024/Env_extensions/' + footprint + 'Footprint_23ed_' + str(year) + '.csv')
    co2_figaro[year] = co2_figaro[year].set_index(['ref_area', 'industry', 'counterpart_area', 'sto'])[['obs_value']].unstack(['sto', 'counterpart_area'])\
        .droplevel(axis=1, level=0).drop(np.nan, axis=1)
    co2_figaro[year].index = [x[0] + '_' + x[1] for x in co2_figaro[year].index.tolist()]
    co2_figaro[year].columns = [x[1] + '_' + str(x[0]) for x in co2_figaro[year].columns.tolist()]

pickle.dump(co2_figaro, open(emissions_filepath + 'Figaro/Figaro_emissions_' + footprint + '_v' + version + '.p', 'wb'))
    
##########
## OECD ##
##########

oecd_data = {}

lookup = pd.read_excel(mrio_filepath + 'ICIO/Ed_2024/ReadMe_ICIO_small.xlsx', sheet_name = 'Area_Activities', header=2)
lookup_country = lookup[['Code', 'countries']].dropna(how='all')\
    .append(lookup[['Column1', 'countries']].dropna(how='any').rename(columns={'Column1':'Code'}))
lookup_industry = lookup[['Code.1', 'Industry']].dropna(how='all')
lookup_fd = ['HFCE', 'NPISH', 'GGFC', 'GFCF', 'INVNT', 'DPABR']

lookup_cols = pd.read_excel(mrio_filepath + 'ICIO/Ed_2024/ReadMe_ICIO_small.xlsx', sheet_name = 'ColItems', header=3, index_col='ID')
lookup_rows = pd.read_excel(mrio_filepath + 'ICIO/Ed_2024/ReadMe_ICIO_small.xlsx', sheet_name = 'RowItems', header=3, index_col='ID')

temp = pd.read_csv(mrio_filepath + 'ICIO/Ed_2024/Env_extensions/DF_MAIN.csv')
temp = temp.loc[(temp['MEASURE'] == 'PROD_GHG') & (temp['UNIT_MEASURE'] == 'T_CO2E')]
temp['REF_AREA'] = temp['REF_AREA'].replace('WXD', 'ROW')

for year in years:
    
    oecd_data[year] = {}

    name = mrio_filepath + 'ICIO/Ed_2024/' + str(year) + '_SML.csv'         
    icio = pd.read_csv(name, index_col=0)
    
    # save fs cats to filter out Y 
    Z_cols = lookup_cols.loc[lookup_cols['Industry/Final demand'].isin(lookup_industry['Code.1']) == True]['Sector code']
    Z_rows = lookup_rows.loc[lookup_rows['Industry/Final demand'].isin(lookup_industry['Code.1']) == True]['Sector code']
    
    Y_cols = cp.copy(lookup_cols)
    Y_cols['Ind'] = [str(x).split('_')[-1] for x in Y_cols['Sector code']]
    Y_cols = Y_cols.loc[Y_cols['Ind'].isin(lookup_fd) == True]['Sector code']

    oecd_data[year][footprint] = temp.loc[(temp['TIME_PERIOD'] == year) &
                                      (temp['REF_AREA'].isin(lookup_country['Code']) == True) &
                                      (temp['ACTIVITY'].isin(lookup_industry['Code.1']) == True)]
    oecd_data[year][footprint]['Sector code'] = oecd_data[year][footprint]['REF_AREA'] + '_' + oecd_data[year][footprint]['ACTIVITY']
    oecd_data[year][footprint] = oecd_data[year][footprint].set_index(['Sector code'])[['OBS_VALUE']]
    oecd_data[year][footprint] = oecd_data[year][footprint]*1000 # adjust unit to match Figaro
    oecd_data[year][footprint].loc[Z_cols]

    oecd_data[year]['Z'] = icio.loc[Z_rows, Z_cols]
    oecd_data[year]['Y'] = icio.loc[Z_rows, Y_cols]

# calculate oecd footprint
co2_oecd = {}
for year in years:
    Z = oecd_data[year]['Z']; Y = oecd_data[year]['Y']; stressor = oecd_data[year][footprint]
    co2_oecd[year] = cef.indirect_footprint_Z(Z, Y, stressor)
    co2_oecd[year].index = pd.MultiIndex.from_arrays([[x.split('_')[0] for x in co2_oecd[year].index], 
                                                      [x.split('_')[1] for x in co2_oecd[year].index]])
    co2_oecd[year].columns = pd.MultiIndex.from_arrays([[x.split('_')[0] for x in co2_oecd[year].columns], 
                                                       [x.split('_')[1] for x in co2_oecd[year].columns]])

    
pickle.dump(co2_oecd, open(emissions_filepath + 'ICIO/ICIO_emissions_' + footprint + '_v' + version + '.p', 'wb'))

############
## Gloria ##
############

"""
In separate script
"""   


    