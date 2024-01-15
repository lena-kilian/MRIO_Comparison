# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:31:51 2023

@author: geolki
"""

# use pymrio https://pymrio.readthedocs.io/en/latest/notebooks/autodownload.html

# conda activate mrio

import pandas as pd
from sys import platform
import calculate_emissions_functions as cef


# set working directory
# make different path depending on operating system
if platform[:3] == 'win':
    wd = 'O://'
else:
    wd = r'/Volumes/a72/' 

# define filepaths
mrio_filepath = wd + 'ESCoE_Project/data/MRIO/'
outputs_filepath = wd + 'UKMRIO_Data/outputs/results_2023/'


years = range(2010, 2019)

############
## Gloria ##
############

readme = mrio_filepath + 'Gloria/GLORIA_ReadMe_057.xlsx'
labels = pd.read_excel(readme, sheet_name=None)

# get lookup to fix labels
lookup = pd.read_excel('O://ESCoE_Project/data/lookups/mrio_lookup_sectors_countries_finaldemand.xlsx', sheet_name=None)

lookup['countries'] = lookup['countries'][['gloria', 'gloria_code_long']].drop_duplicates().dropna()
lookup['countries']['gloria_combo'] = lookup['countries']['gloria'] + ' (' + lookup['countries']['gloria_code_long'] + ') '

lookup['sectors'] = lookup['sectors'][['gloria']].drop_duplicates().dropna()

# fix Z labels
t_cats = pd.DataFrame(labels['Sequential region-sector labels']['Sequential_regionSector_labels']).drop_duplicates(); t_cats.columns = ['label']
temp_c = []
for cs in t_cats['label']:
    a = False
    for item in cs.split('('):
        if item.split(')')[0] in lookup['countries']['gloria_code_long'].tolist():
            a = True
            c = cs.split(item.split(')')[0])[0] + item.split(')')[0] + ')'
            temp_c.append(c)
    if a == False:
        temp_c.append('NA')
        
if 'NA' in temp_c:
    print('Missing coutry labels')
    raise SystemExit
        
t_cats['country_full'] = temp_c
t_cats['country'] = [x.split('(')[-1][:-1] for x in t_cats['country_full']]
temp_s = []
for i in range(len(t_cats)):
    temp = t_cats.iloc[i, :]
    temp_s.append(temp['label'].replace(temp['country_full'], ''))
t_cats['sector'] = temp_s

# fix final demand labels
fd_cats = pd.DataFrame(labels['Sequential region-sector labels']['Sequential_finalDemand_labels'].dropna(how='all', axis=0)); fd_cats.columns = ['label']

temp_c = []
for cs in fd_cats['label']:
    a = False
    for item in cs.split('('):
        if item.split(')')[0] in lookup['countries']['gloria_code_long'].tolist():
            a = True
            c = cs.split(item.split(')')[0])[0] + item.split(')')[0] + ')'
            temp_c.append(c)
    if a == False:
        temp_c.append('NA')
        
if 'NA' in temp_c:
    print('Missing coutry labels')
    raise SystemExit

fd_cats['country_full'] = temp_c
fd_cats['country'] = [x.split('(')[-1][:-1] for x in fd_cats['country_full']]
temp_s = []
for i in range(len(fd_cats)):
    temp = fd_cats.iloc[i, :]
    temp_s.append(temp['label'].replace(temp['country_full'], ''))
fd_cats['fd'] = temp_s

# keep only industries
t_cats['ind'] = t_cats['label'].str[-8:]
industries = t_cats.loc[t_cats['ind'] == 'industry']
products = t_cats.loc[t_cats['ind'] != 'industry']

# make index labels
z_idx = pd.MultiIndex.from_arrays([t_cats['country'], t_cats['sector']])
industry_idx = pd.MultiIndex.from_arrays([industries['country'], industries['sector']])
product_idx = pd.MultiIndex.from_arrays([products['country'], products['sector']])
y_cols = pd.MultiIndex.from_arrays([fd_cats['country'], fd_cats['fd']])

sat_rows = labels['Satellites']['Sat_indicator']
stressor_cat = "'co2_excl_short_cycle_org_c_total_EDGAR_consistent'"

# clear space in variable explorer to clear RAM
del t_cats, temp_c, cs, a, c, temp_s, i, temp, fd_cats, industries, products, item
    
check_uk_gloria = {}

for year in years:

    if year < 2017:
        date_var = '20230314'
    else:
        date_var = '20230315'
    
    z_filepath = (mrio_filepath + 'Gloria/Main/' + date_var + '_120secMother_AllCountries_002_T-Results_' + str(year) + '_057_Markup001(full).csv') 
    y_filepath = (mrio_filepath + 'Gloria/Main/' + date_var + '_120secMother_AllCountries_002_Y-Results_' + str(year) + '_057_Markup001(full).csv') 
    co2_filepath = (mrio_filepath + 'Gloria/Satellite_Accounts/20230727_120secMother_AllCountries_002_TQ-Results_' + str(year) + '_057_Markup001(full).csv') 
    
    Z = pd.read_csv(z_filepath, header=None, index_col=None)
    Z.index = z_idx; Z.columns = z_idx
    S = Z.loc[industry_idx, product_idx]
    U = Z.loc[product_idx, industry_idx]
    del Z # remove Z to clear memory
    
    Y = pd.read_csv(y_filepath, header=None, index_col=None)
    Y.index = z_idx; Y.columns = y_cols
    Y = Y.loc[product_idx]
    
    stressor = pd.read_csv(co2_filepath, header=None, index_col=None)
    stressor.index = sat_rows; stressor.columns = z_idx
    stressor = stressor.loc[stressor_cat, industry_idx]
    
    print('Data loaded for ' + str(year))

    # calculate gloria footprint
    co2_gloria = cef.indirect_footprint_SUT(S, U, Y, stressor)

    print('Footprint calculated for ' + str(year))
    
    co2_gloria.to_csv('O:/ESCoE_Project/data/Emissions/Gloria/CO2_' + str(year) + '.csv')
    
    print('Footprint saved for ' + str(year))

print('Gloria done')