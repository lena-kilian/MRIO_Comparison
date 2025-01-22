# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:31:51 2023

@author: geolki
"""

import pandas as pd
from sys import platform
import calculate_emissions_functions_gloria as cef_g
import calculate_emissions_functions as cef
import pickle
import copy as cp
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

if footprint == 'co2':
    mrio_list = ['exio', 'figaro', 'gloria']
elif footprint =='ghg':
    mrio_list = ['exio', 'figaro', 'oecd', 'gloria']

lookup = pd.read_excel(wd + 'ESCoE_Project/data/lookups/mrio_lookup_sectors_countries_finaldemand.xlsx', sheet_name=None) 

years = range(1990, 2023)

############
## Direct ##
############

# define filepaths
ons_filepath = wd + 'UKMRIO_Data/data/raw/ONS/'

file = ons_filepath + 'ONS environmental accounts/2025/provisionalatmoshpericemissionsghg.xlsx'
temp_uk_ghg_sectors = np.transpose(pd.read_excel(file, 'GHG total', usecols='A:EB', index_col=0, header=0, nrows=33, skiprows = 46))

file = ons_filepath + 'Analytical tables/sectorconc_112.xlsx'
uk_emissions_conc = pd.read_excel(file, 'emissions',index_col = 0)

uk_ghg_sectors = {}
for year in years:
    temp_ghg = np.dot(np.transpose(temp_uk_ghg_sectors.loc[:,year]),uk_emissions_conc)
    uk_ghg_sectors[year] = temp_ghg

uk_ghg_direct = temp_uk_ghg_sectors.iloc[129:131,0:33]

pickle.dump(uk_ghg_direct, open(emissions_filepath + 'uk_direct.p', 'wb'))

print('Direct Done')


############
## Figaro ##
############

co2_figaro_ind = {}; co2_figaro_prod = {}

# make lookup
lookup_fd = lookup['final_demand'][['figaro_code', 'combined_name']].dropna(how='any').drop_duplicates()
lookup_fd = dict(zip(lookup_fd['figaro_code'], lookup_fd['combined_name']))

# get figaro footprint
for year in range(2010, 2023):

    S = pd.read_csv(mrio_filepath + 'Figaro/Ed_2024/matrix_eu-ic-supply_24ed_' + str(year) + '.csv', index_col=0).T
    S.columns = pd.MultiIndex.from_arrays([[x.split('_')[0] for x in S.columns],  ['_'.join(x.split('_')[1:]) for x in S.columns]])
    S.index = pd.MultiIndex.from_arrays([[x.split('_')[0] for x in S.index], ['_'.join(x.split('_')[1:]) for x in S.index]])
    
    U_all = pd.read_csv(mrio_filepath + 'Figaro/Ed_2024/matrix_eu-ic-use_24ed_' + str(year) + '.csv', index_col=0)
    U_all.columns = pd.MultiIndex.from_arrays([[x.split('_')[0] for x in U_all.columns], ['_'.join(x.split('_')[1:]) for x in U_all.columns]])
    U_all.index = pd.MultiIndex.from_arrays([[x.split('_')[0] for x in U_all.index], ['_'.join(x.split('_')[1:]) for x in U_all.index]])
    
    U = U_all.loc[S.columns, S.index]
 
    y_cols = []
    for item in list(lookup_fd.keys()):
        if item in U_all.columns.levels[1]:
            y_cols.append(item)
            
    Y = U_all.swaplevel(axis=1)[y_cols].swaplevel(axis=1).loc[S.columns,:]
    
    stressor = pd.read_csv(mrio_filepath + 'Figaro/Ed_2024/Env_extensions/' + footprint + 'Footprint_23ed_' + str(year) + '.csv')
    stressor = stressor.groupby(['ref_area', 'industry'])['obs_value'].sum().loc[S.index]

    # calculate footprint
    emissions_ind, emissions_prod = cef.indirect_footprint_SUT(S, U, Y, stressor)
    
    # save
    co2_figaro_ind[year] = emissions_ind
    co2_figaro_prod[year] = emissions_prod

pickle.dump(co2_figaro_ind, open(emissions_filepath + 'Figaro/Figaro_industry_' + footprint + '_v' + version + '_uk.p', 'wb'))
pickle.dump(co2_figaro_prod, open(emissions_filepath + 'Figaro/Figaro_products_' + footprint + '_v' + version + '_uk.p', 'wb'))

print('Figaro Done')

############
## Gloria ##
############

co2_gloria_prod = {}; co2_gloria_ind = {}

# make lookup for aggregation
lookup_country = lookup['countries'][['gloria_code', 'combined_name']].dropna(how='any').drop_duplicates()
lookup_country = dict(zip(lookup_country['gloria_code'], lookup_country['combined_name']))

temp = lookup['sectors'][['gloria', 'combined_name']].dropna(how='any').drop_duplicates()
temp2 = cp.copy(temp)
temp2['gloria'] = ' ' + temp2['gloria'] + ' industry'
temp['gloria'] = ' ' + temp['gloria'] + ' product'
lookup_sectors = temp.append(temp2)
lookup_sectors = dict(zip(lookup_sectors['gloria'], lookup_sectors['combined_name']))

lookup_fd = lookup['final_demand'][['gloria', 'combined_name']].dropna(how='any').drop_duplicates()
lookup_fd = dict(zip(lookup_fd['gloria'], lookup_fd['combined_name']))

# read config file to get filenames
config_file= wd + 'ESCoE_Project/data/MRIO/Gloria/config_large.cfg'
gloria_filepath, outdir, lookup_filepath, labels_fname, lookup_fname, Z_fname, Y_fname, co2_fname, gloria_version = cef_g.read_config(config_file)

z_idx, industry_idx, product_idx, iix,pix,y_cols, sat_rows = cef_g.get_metadata_indices(gloria_filepath, lookup_filepath, labels_fname, lookup_fname)

# use this to extract correct row from stressor dataset below. Only one row from this DF is needed in the analysis
if footprint == 'ghg':
    stressor_cat = "'GHG_total_EDGAR_consistent'" 
elif footprint == 'co2':
    stressor_cat = "'co2_excl_short_cycle_org_c_total_EDGAR_consistent'"

# JAC work out which row stressor_cat is on
stressor_row = pd.Index(sat_rows).get_loc(stressor_cat)

# define sample year, normally this is: range(2010, 2019)
# here years is now determined from inputs,
# it used to be a range(2010, 2019). In future work this will likley be range(2001, 2023)
for year in years:

    # set up filepaths
    # file name changes from 2017, so define this here
    split=Z_fname.split('%')
    if len(split)>1:
        z_filepath=gloria_filepath+gloria_version+split[0]+str(year)+split[1]
    else:
        z_filepath=gloria_filepath+gloria_version+Z_fname

    split=Y_fname.split('%')
    if len(split)>1:
        y_filepath=gloria_filepath+gloria_version+split[0]+str(year)+split[1]
    else:
        y_filepath=gloria_filepath+gloria_version+Y_fname
    
    if year < 1997:
        z_filepath = z_filepath.replace('20240111', '20240110')
        y_filepath = y_filepath.replace('20240111', '20240110')

    split=co2_fname.split('%')
    if len(split)>1:
        co2_filepath=gloria_filepath+gloria_version+'Env_extensions/'+split[0]+str(year)+split[1]
    else:
        co2_filepath=gloria_filepath+gloria_version+'Env_extensions/'+co2_fname

    S, U, Y, stressor = cef_g.read_data_new(z_filepath, y_filepath, co2_filepath, iix, pix, industry_idx, product_idx, y_cols, stressor_row)

    emissions_ind = cef_g.indirect_footprint_SUT_ind(S, U, Y, stressor).T
    emissions_prod = cef_g.indirect_footprint_SUT_prod(S, U, Y, stressor).T

    # aggregate industries and products
    cols0 = [x[0] for x in emissions_prod.columns.tolist()]
    cols1 = []
    for x in emissions_prod.columns.tolist():
        temp = x[1]
        if temp[0] == ' ':
            cols1.append(temp[1:])
        else:
            cols1.append(temp)
    
    emissions_prod.columns = pd.MultiIndex.from_arrays([cols0, cols1])
    emissions_prod = emissions_prod.rename(columns=lookup_country, index=lookup_country).rename(columns=lookup_fd, index=lookup_sectors).sum(axis=0, level=[0,1]).sum(axis=1, level=[0,1])

    # aggregate industries and products
    cols0 = [x[0] for x in emissions_ind.columns.tolist()]
    cols1 = []
    for x in emissions_ind.columns.tolist():
        temp = x[1]
        if temp[0] == ' ':
            cols1.append(temp[1:])
        else:
            cols1.append(temp)
    emissions_ind.columns = pd.MultiIndex.from_arrays([cols0, cols1])
    emissions_ind = emissions_ind.rename(columns=lookup_country, index=lookup_country).rename(columns=lookup_fd, index=lookup_sectors).sum(axis=0, level=[0,1]).sum(axis=1, level=[0,1])
    
    # save
    co2_gloria_prod[year] = emissions_prod
    co2_gloria_ind[year] = emissions_ind
    print('GLORIA ' + str(year))

pickle.dump(co2_gloria_ind, open(emissions_filepath + 'Gloria/Gloria_industry_' + footprint + '_v' + version + '_uk.p', 'wb'))
pickle.dump(co2_gloria_prod, open(emissions_filepath + 'Gloria/Gloria_products_' + footprint + '_v' + version + '_uk.p', 'wb'))

print('Gloria Done')