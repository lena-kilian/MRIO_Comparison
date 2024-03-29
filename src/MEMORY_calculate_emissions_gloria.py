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
from datetime import datetime
import psutil
import inspect
import numpy as np

stats = pd.DataFrame()
start_time_all = datetime.now()

# add timestamp
def add_time_mem(stats, start_time_all, year):
    temp = pd.DataFrame(psutil.virtual_memory()).T
    temp.columns=['total', 'available', 'percent', 'used', 'free']
    temp['duration'] = datetime.now() - start_time_all
    temp['line_no'] = int(inspect.getframeinfo(inspect.stack()[1][0]).lineno)
    temp['year'] = year
    stats = stats.append(temp)
    print(stats)
    return(stats)

stats = add_time_mem(stats, start_time_all, 0)

# set working directory
# make different path depending on operating system
if platform[:3] == 'win':
    wd = 'O://'
else:
    wd = r'/Volumes/a72/' 

# define filepaths
mrio_filepath = wd + 'ESCoE_Project/data/MRIO/'
outputs_filepath = wd + 'UKMRIO_Data/outputs/results_2023/'


years = [2016] #range(2016, 2019)

############
## Gloria ##
############

readme = mrio_filepath + 'Gloria/GLORIA_ReadMe_057_small.xlsx'
labels = pd.read_excel(readme, sheet_name=None)

# get lookup to fix labels
lookup = pd.read_excel('O://ESCoE_Project/data/lookups/mrio_lookup_sectors_countries_finaldemand_small.xlsx', sheet_name=None)

lookup['countries'] = lookup['countries'][['gloria', 'gloria_code']].drop_duplicates().dropna()

lookup['countries']['gloria_combo'] = lookup['countries']['gloria'] + ' (' + lookup['countries']['gloria_code'] + ') '

lookup['sectors'] = lookup['sectors'][['gloria']].drop_duplicates().dropna()

# fix Z labels
t_cats = pd.DataFrame(labels['Sequential region-sector labels']['Sequential_regionSector_labels']).drop_duplicates(); t_cats.columns = ['label']
temp_c = []
for cs in t_cats['label']:
    a = False
    for item in cs.split('('):
        if item.split(')')[0] in lookup['countries']['gloria_code'].tolist():
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

stats = add_time_mem(stats, start_time_all, 0)

temp_c = []
for cs in fd_cats['label']:
    a = False
    for item in cs.split('('):
        if item.split(')')[0] in lookup['countries']['gloria_code'].tolist():
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

stats = add_time_mem(stats, start_time_all, 0)

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
times = {}
start_time_years = datetime.now()

stats = add_time_mem(stats, start_time_all, 0)

for year in years:

    if year < 2017:
        date_var = '20230314'
    else:
        date_var = '20230315'
    
    z_filepath = (mrio_filepath + 'Gloria/Z_small.csv')
    y_filepath = (mrio_filepath + 'Gloria/Y_small.csv') 
    co2_filepath = (mrio_filepath + 'Gloria/stressor_small.csv') 
    stats = add_time_mem(stats, start_time_all, year)
    
    
    Z = pd.read_csv(z_filepath, header=None, index_col=None)
    Z.index = z_idx; Z.columns = z_idx
    
    S = Z.loc[industry_idx, product_idx]
    U = Z.loc[product_idx, industry_idx]
    stats = add_time_mem(stats, start_time_all, year)
    del Z # remove Z to clear memory
    stats = add_time_mem(stats, start_time_all, year)
    
    Y = pd.read_csv(y_filepath, header=None, index_col=None)
    Y.index = z_idx; Y.columns = y_cols    
    Y = Y.loc[product_idx]
    
    stats = add_time_mem(stats, start_time_all, year)
    
    stressor = pd.read_csv(co2_filepath, header=None, index_col=None)
    stressor.index = sat_rows; stressor.columns = z_idx
    stressor = stressor.loc[stressor_cat, industry_idx]

    stats = add_time_mem(stats, start_time_all, year)
    
    print('Data loaded for ' + str(year))
    stats = add_time_mem(stats, start_time_all, year)
    
    # save column names
    z_idx = pd.MultiIndex.from_arrays([[x[0] for x in S.columns.tolist()] + [x[0] for x in U.columns.tolist()],
                                       [x[1] for x in S.columns.tolist()] + [x[1] for x in U.columns.tolist()]])
    u_cols = U.columns.tolist()

    # calculate gloria footprint
    Z = cef.make_Z_from_S_U(S, U) 
    stats = add_time_mem(stats, start_time_all, year)
    del S, U # remove S and U to clear memory
    stats = add_time_mem(stats, start_time_all, year)
    bigY = np.zeros(shape = [np.size(Y, 0)*2, np.size(Y, 1)])
    stats = add_time_mem(stats, start_time_all, year)
    
    footprint = np.zeros(shape = bigY.shape).T
    stats = add_time_mem(stats, start_time_all, year)
    
    bigY[np.size(Y, 0):np.size(Y, 0)*2, 0:] = Y     
    stats = add_time_mem(stats, start_time_all, year)
    x = cef.make_x(Z, bigY)
    stats = add_time_mem(stats, start_time_all, year)
    L = cef.make_L(Z, x)
    stats = add_time_mem(stats, start_time_all, year)
    bigstressor = np.zeros(shape = [np.size(Y, 0)*2, 1])
    stats = add_time_mem(stats, start_time_all, year)
    bigstressor[:np.size(Y, 0), 0] = np.array(stressor)
    stats = add_time_mem(stats, start_time_all, year)
    e = np.sum(bigstressor, 1)/x
    stats = add_time_mem(stats, start_time_all, year)
    eL = np.dot(e, L)
    stats = add_time_mem(stats, start_time_all, year)
    
    for a in range(np.size(Y, 1)):
        footprint[a] = np.dot(eL, np.diag(bigY[:, a]))
        stats = add_time_mem(stats, start_time_all, year)
    
    stats = add_time_mem(stats, start_time_all, year)
    footprint = pd.DataFrame(footprint, index=Y.columns, columns=z_idx)
    footprint = footprint[u_cols]
    stats = add_time_mem(stats, start_time_all, year)
    
    print('Footprint calculated for ' + str(year))
    
    footprint.to_csv('O:/ESCoE_Project/data/Emissions/Gloria/MEMORY_TEST_CO2_' + str(year) + '.csv')
    
    print('Footprint saved for ' + str(year))
    stats = add_time_mem(stats, start_time_all, year)

print('Gloria done')

stats = add_time_mem(stats, start_time_all, 9999)