# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:31:51 2023

@author: geolki
"""

import pandas as pd
from sys import platform
import calculate_emissions_functions as cef
import pickle
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
footprint = 'ghg' # 'co2' #

years = range(2010, 2022)
mrio_list = ['exio394']

lookup = pd.read_excel(wd + 'ESCoE_Project/data/lookups/mrio_lookup_sectors_countries_finaldemand.xlsx', sheet_name=None) 

stressor_sums = {}; 
for item in mrio_list:
    stressor_sums[item] = pd.DataFrame(index=lookup['sectors']['combined_name'].unique())

####################
## Exiobase 3.9.6 ##
####################

co2_exio394_prod = {}; co2_exio394_ind = {}

stressor_var = 'GHG emissions AR5 (GWP100)|kg CO2 eq.||GWP100 (IPCC, 2010)'


# make lookup
lookup_country = lookup['countries'][['exio_code', 'combined_name']].dropna(how='any').drop_duplicates()
lookup_country = dict(zip(lookup_country['exio_code'], lookup_country['combined_name']))

lookup_sectors = lookup['sectors'][['exio', 'combined_name']].dropna(how='any').drop_duplicates()
lookup_sectors = dict(zip(lookup_sectors['exio'], lookup_sectors['combined_name']))

lookup_fd = lookup['final_demand'][['exio', 'combined_name']].dropna(how='any').drop_duplicates()
lookup_fd = dict(zip(lookup_fd['exio'], lookup_fd['combined_name']))

co2_raw = {}

for year in years:
    
    exio394_data = {}
                  
    filepath = wd + 'EXIOBASE/3.9.4/MRSUT_' + str(year) + '/'
    filepath_emissions = wd + 'EXIOBASE/3.9.6_processed_GHG/' + str(year) + '.csv'
     
    exio394_data['S'] = pd.read_csv(filepath + 'supply.csv', sep='\t', header = [0, 1], index_col = [0, 1]).T
    exio394_data['U'] = pd.read_csv(filepath + 'use.csv', sep='\t', header = [0, 1], index_col = [0, 1])
    exio394_data['Y'] = pd.read_csv(filepath + 'final_demand.csv', sep='\t', header = [0, 1], index_col = [0, 1])
    exio394_data['co2'] = pd.DataFrame(pd.read_csv(filepath_emissions, index_col=0, header=[0, 1]).loc[stressor_var, :])
    
    co2_raw[year] = exio394_data['co2']
    # calculate exio footprint

    S = exio394_data['S']; U = exio394_data['U']; Y = exio394_data['Y']; stressor = exio394_data['co2']
    
    # aggregate industries and products
    U = U.loc[S.columns, S.index]
    Y = Y.loc[S.columns, :]
    stressor = stressor.loc[S.index, stressor_var] / 1000000
    
    # save for reference
    stressor_sums['exio394'][year] = stressor.rename(index=lookup_sectors).sum(axis=0, level=1) 
    
    emissions_ind, emissions_prod = cef.indirect_footprint_SUT(S, U, Y, stressor)
    # save as csv
    emissions_ind.to_csv('O:/ESCoE_Project/data/Emissions/Exiobase/Exiobase_' + footprint + '_industries_' + str(year) + '.csv')
    emissions_prod.to_csv('O:/ESCoE_Project/data/Emissions/Exiobase/Exiobase_' + footprint + '_products_' + str(year) + '.csv')
    # aggregate industries and products
    emissions_ind = emissions_ind.T.rename(columns=lookup_country, index=lookup_country).rename(columns=lookup_fd, index=lookup_sectors).sum(axis=0, level=[0,1]).sum(axis=1, level=[0,1])
    emissions_prod = emissions_prod.T.rename(columns=lookup_country, index=lookup_country).rename(columns=lookup_fd, index=lookup_sectors).sum(axis=0, level=[0,1]).sum(axis=1, level=[0,1])
    
    # save
    co2_exio394_ind[year] = emissions_ind
    co2_exio394_prod[year] = emissions_prod
    
    print(year)

pickle.dump(co2_exio394_ind, open(emissions_filepath + 'Exiobase/Exiobase_industry_' + footprint + '394_v' + version + '_agg_after.p', 'wb'))
pickle.dump(co2_exio394_prod, open(emissions_filepath + 'Exiobase/Exiobase_products_' + footprint + '394_v' + version + '_agg_after.p', 'wb'))

print('Exiobase Done')

    
##############
## Save all ##
##############

co2_all_prod = {}
co2_all_ind = {}
for item in mrio_list:
    co2_all_prod[item] = eval('co2_' + item + '_prod')
    co2_all_ind[item] = eval('co2_' + item + '_ind')
    
    for year in years:
        print(year, np.round(co2_all_ind[item][year].sum().sum()), np.round(co2_raw[year].sum().sum()))

pickle.dump(co2_all_prod, open(emissions_filepath + 'Emissions_products_' + footprint + '_exio394_agg_after.p', 'wb'))
pickle.dump(co2_all_ind, open(emissions_filepath + 'Emissions_industry_' + footprint + '_exio394_agg_after.p', 'wb'))

pickle.dump(stressor_sums, open(emissions_filepath + 'Industry_emissions_' + footprint + '_exio394_from_stressor.p', 'wb'))
