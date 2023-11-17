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
import itertools


# set working directory
# make different path depending on operating system
if platform[:3] == 'win':
    wd = 'O://'
else:
    wd = r'/Volumes/a72/' 

# define filepaths
mrio_filepath = wd + 'ESCoE_Project/data/MRIO/'
outputs_filepath = wd + 'UKMRIO_Data/outputs/results_2023/'

#years = range(2010, 2019)
years = [2018]

##############
## EXIOBASE ##
##############

exio_data = {}

for year in years:
    
    exio_data[year] = {}
                  
    filepath = wd + 'UKMRIO_Data/EXIOBASE/IOT/IOT_' + str(year) + '_ixi/'
            
    exio_data[year]['Z'] = pd.read_csv(filepath + 'Z.txt', sep='\t', header = [0, 1], index_col = [0, 1])
    exio_data[year]['Y'] = pd.read_csv(filepath + 'Y.txt', sep='\t', header = [0, 1], index_col = [0, 1])
    exio_data[year]['co2'] = pd.DataFrame(pd.read_csv(filepath + 'satellite/F.txt', sep='\t', index_col=0, header=[0, 1])\
        .loc['CO2 - combustion - air', :])
    print(year)

# calculate exio footprint
co2_exio = {}
for year in years:
    Z = exio_data[year]['Z']; Y = exio_data[year]['Y']; stressor = exio_data[year]['co2']
    co2_exio[year] = cef.indirect_footprint(Z, Y, stressor)
    print(year)
    
# check if UK result makes sense    
check_uk_exio = {}
for year in years:
    check_uk_exio[year] = co2_exio[year]['GB'].sum(0) / 1000000000
    print(year)
      
############
## Figaro ##
############

co2_figaro = {}

for year in years:
    url = ('https://ec.europa.eu/eurostat/documents/51957/14504006/CO2_footprints_' + str(year) + 
           '.csv/22bc76fb-1860-2313-5af6-691667ba0b03?t=1671543343180')
    co2_figaro[year] = pd.read_csv(url).set_index(['ref_area', 'industry', 'counterpart_area', 'sto'])[['obs_value']]\
        .unstack(['counterpart_area', 'sto']).fillna(0).droplevel(axis=1, level=0)
    print(year)
    
# check if UK result makes sense    
check_uk_figaro = {}
for year in years:
    check_uk_figaro[year] = co2_figaro[year]['GB'].sum(0) / 1000
    print(year)
    
##########
## OECD ##
##########

oecd_data = {}

for year in years:
    
    oecd_data[year] = {}

    name = mrio_filepath + 'ICIO/ICIO2021_' + str(year) + '.csv'         
    icio = pd.read_csv(name, index_col=0)
    
    # save fs cats to filter out Y 
    cut_off = icio.columns.tolist().index('AUS_HFCE')
    fd = icio.columns.tolist()[cut_off:]; fd.remove('TOTAL')
    fd_cats = []
    for item in fd:
        if item.split('_')[1] not in fd_cats:
            fd_cats.append(item.split('_')[1])
    
    icio.index = [item.replace('CN1', 'CHN').replace('CN2', 'CHN').replace('MX1', 'MEX').replace('MX2', 'MEX') for item in icio.index]
    icio.columns = [item.replace('CN1', 'CHN').replace('CN2', 'CHN').replace('MX1', 'MEX').replace('MX2', 'MEX') for item in icio.columns]
    
    icio = icio.reset_index().groupby('index').sum().T.reset_index().groupby('index').sum().T
    
    oecd_data[year]['co2'] = pd.read_csv( mrio_filepath + 'ICIO/Env_extensions/Extension_data_' + str(year) + '_PROD_CO2_WLD.csv')
    countries = oecd_data[year]['co2'][['COU']].drop_duplicates()['COU'].tolist() # save to filter out Y 
    oecd_data[year]['co2']['index'] = oecd_data[year]['co2']['COU'] + '_' + oecd_data[year]['co2']['IND'].str[1:]
    oecd_data[year]['co2'] = oecd_data[year]['co2'].set_index('index')[['VALUE']]

    fd = list(itertools.product(countries, fd_cats))
    fd = [x[0] + '_' + x[1] for x in fd]
    oecd_data[year]['Z'] = icio.loc[oecd_data[year]['co2'].index.tolist(), oecd_data[year]['co2'].index.tolist()]
    oecd_data[year]['Y'] = icio.loc[oecd_data[year]['co2'].index.tolist(), fd]
    
    print(year)

# calculate oecd footprint
co2_oecd = {}
for year in years:
    Z = oecd_data[year]['Z']; Y = oecd_data[year]['Y']; stressor = oecd_data[year]['co2']
    co2_oecd[year] = cef.indirect_footprint(Z, Y, stressor)
    co2_oecd[year].index = pd.MultiIndex.from_arrays([[x.split('_')[0] for x in co2_oecd[year].index], 
                                                      [x.split('_')[1] for x in co2_oecd[year].index]])
    co2_oecd[year].columns = pd.MultiIndex.from_arrays([[x.split('_')[0] for x in co2_oecd[year].columns], 
                                                       [x.split('_')[1] for x in co2_oecd[year].columns]])
    
    print(year)
    
# check if UK result makes sense    
check_uk_oecd = {}
for year in years:
    check_uk_oecd[year] = co2_oecd[year]['GBR'].sum(0) 
    print(year)

############
## Gloria ##
############

gloria_data = {}

readme = mrio_filepath + 'Gloria/GLORIA_ReadMe_057.xlsx'
labels = pd.read_excel(readme, sheet_name=None)

t_cats = labels['Sequential region-sector labels']['Sequential_regionSector_labels'].tolist()
z_idx = pd.MultiIndex.from_arrays([[x.split('(')[1].split(')')[0] for x in t_cats],
                                   [x.split(') ')[1] for x in t_cats]])

fd_cats = labels['Sequential region-sector labels']['Sequential_finalDemand_labels'].dropna(how='all', axis=0).tolist()
y_cols = pd.MultiIndex.from_arrays([[x.split('(')[1].split(')')[0] for x in fd_cats],
                                    [x.split(') ')[1] for x in fd_cats]])

sat_rows = labels['Satellites']['Sat_indicator']

stressor_cat = "'co2_excl_short_cycle_org_c_total_EDGAR_consistent'"
    
for year in years:
    
    gloria_data[year] = {}
    
    z_filepath = (mrio_filepath + 'Gloria/Main/20230315_120secMother_AllCountries_002_T-Results_' + str(year) + '_057_Markup001(full).csv') 
    y_filepath = (mrio_filepath + 'Gloria/Main/20230315_120secMother_AllCountries_002_Y-Results_' + str(year) + '_057_Markup001(full).csv') 
    co2_filepath = (mrio_filepath + 'Gloria/Satellite_Accounts/20230727_120secMother_AllCountries_002_TQ-Results_' + str(year) + '_057_Markup001(full).csv') 
    
    gloria_data[year]['Z'] = pd.read_csv(z_filepath, header=None, index_col=None)
    gloria_data[year]['Z'].index = z_idx; gloria_data[year]['Z'].columns = z_idx
    
    gloria_data[year]['Y'] = pd.read_csv(y_filepath, header=None, index_col=None)
    gloria_data[year]['Y'].index = z_idx; gloria_data[year]['Y'].columns = y_cols
    
    gloria_data[year]['co2'] = pd.read_csv(co2_filepath, header=None, index_col=None)
    gloria_data[year]['co2'].index = sat_rows; gloria_data[year]['co2'].columns = z_idx
    gloria_data[year]['co2'] = gloria_data[year]['co2'][year].loc[stressor_cat,:]
    
    print(year)
 
# calculate gloria footprint
co2_gloria = {}
for year in years:
    Z = gloria_data[year]['Z']; Y = gloria_data[year]['Y']; stressor = gloria_data[year]['co2']
    co2_gloria[year] = cef.indirect_footprint(Z, Y, stressor)
    co2_gloria[year].index = pd.MultiIndex.from_arrays([[x.split('_')[0] for x in co2_gloria[year].index], 
                                                       [x.split('_')[1] for x in co2_oecd[year].index]])
    co2_oecd[year].columns = pd.MultiIndex.from_arrays([[x.split('_')[0] for x in co2_oecd[year].columns], 
                                                       [x.split('_')[1] for x in co2_oecd[year].columns]])
    
    print(year)
     
# check if UK result makes sense    
check_uk_oecd = {}
for year in years:
    check_uk_oecd[year] = co2_oecd[year]['GBR'].sum(0)
    print(year)
    