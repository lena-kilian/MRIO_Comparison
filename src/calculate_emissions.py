# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:31:51 2023

@author: geolki
"""

# use pymrio https://pymrio.readthedocs.io/en/latest/notebooks/autodownload.html

# conda activate mrio

import pymrio
import os
import pandas as pd
import zipfile
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
data_filepath = wd + 'UKMRIO_Data/data/'
outputs_filepath = wd + 'UKMRIO_Data/outputs/results_2023/'


years = range(2010, 2019)

currency_usd_eur = pd.read_excel(wd + 'UKMRIO_Data/ICIO/ReadMe_ICIO2021_CSV.xlsx', sheet_name='NCU-USD', index_col=[1], header=0).loc['AUT', :][years].astype(float) # conversion rates from ICIO table

co2_props = pd.read_excel(wd + 'UKMRIO_Data/data/processed/uk energy/UKenergy2023.xlsx', sheet_name='co2_props', header = 0, index_col=0)

co2_direct = pd.read_excel(outputs_filepath + 'uk_co2_direct.xlsx',sheet_name=None, index_col=0)

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

# calculate exio footprint
co2_exio = {}
for year in years:
    Z = exio_data[year]['Z']; Y = exio_data[year]['Y']; stressor = exio_data[year]['co2']
    co2_exio[year] = cef.indirect_footprint(Z, Y, stressor)
    
# check if UK result makes sense    
check_uk_exio = {}
for year in years:
    check_uk_exio[year] = co2_exio[year]['GB'].sum(0)
      
############
## Figaro ##
############

figaro_data = {}

for year in years:
    
    figaro_data[year] = {}

    figaro_data[year]['S'] = pd.read_csv(wd + 'UKMRIO_Data/Figaro/matrix_eu-ic-supply_' + str(year) + '.csv', index_col=0)
    use_temp = pd.read_csv(wd + 'UKMRIO_Data/Figaro/matrix_eu-ic-use_' + str(year) + '.csv', index_col=0)
    
    figaro_data[year]['S'].columns = [x.replace('L68', 'L').replace('_CPA', '') for x in figaro_data[year]['S'].columns]
    figaro_data[year]['S'].index = [x.replace('L68', 'L').replace('_CPA', '') for x in figaro_data[year]['S'].index]
    use_temp.columns = [x.replace('L68', 'L').replace('_CPA', '') for x in use_temp.columns]
    use_temp.index = [x.replace('L68', 'L').replace('_CPA', '') for x in use_temp.index]
    
    figaro_data[year]['U'] = use_temp.loc[figaro_data[year]['S'].index, figaro_data[year]['S'].columns]

    figaro_data[year]['Y'] = use_temp.loc[figaro_data[year]['S'].index, :].drop(figaro_data[year]['S'].columns.tolist(), axis=1)


##########
## OECD ##
##########

oecd_data = {}

for year in years:
    
    oecd_data[year] = {}

    name = wd + 'UKMRIO_Data/ICIO/ICIO2021_' + str(year) + '.csv'         
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
    
    oecd_data[year]['co2'] = pd.read_csv( wd + 'UKMRIO_Data/ICIO/Env_extensions/Extension_data_' + str(year) + '_PROD_CO2_WLD.csv')
    countries = oecd_data[year]['co2'][['COU']].drop_duplicates()['COU'].tolist() # save to filter out Y 
    oecd_data[year]['co2']['index'] = oecd_data[year]['co2']['COU'] + '_' + oecd_data[year]['co2']['IND'].str[1:]
    oecd_data[year]['co2'] = oecd_data[year]['co2'].set_index('index')[['VALUE']]

    fd = list(itertools.product(countries, fd_cats))
    fd = [x[0] + '_' + x[1] for x in fd]
    oecd_data[year]['Z'] = icio.loc[oecd_data[year]['co2'].index.tolist(), oecd_data[year]['co2'].index.tolist()]
    oecd_data[year]['Y'] = icio.loc[oecd_data[year]['co2'].index.tolist(), fd]

# calculate exio footprint
co2_oecd = {}
for year in years:
    Z = oecd_data[year]['Z']; Y = oecd_data[year]['Y']; stressor = oecd_data[year]['co2']
    co2_oecd[year] = cef.indirect_footprint(Z, Y, stressor)
    co2_oecd[year].index = pd.MultiIndex.from_arrays([[x.split('_')[0] for x in co2_oecd[year].index], 
                                                      [x.split('_')[1] for x in co2_oecd[year].index]])
    co2_oecd[year].columns = pd.MultiIndex.from_arrays([[x.split('_')[0] for x in co2_oecd[year].columns], 
                                                       [x.split('_')[1] for x in co2_oecd[year].columns]])
    
# check if UK result makes sense    
check_uk_oecd = {}
for year in years:
    check_uk_oecd[year] = co2_oecd[year]['GBR'].sum(0)

############
## Gloria ##
############

gloria_folder = wd + 'UKMRIO_Data/Gloria'

os.chdir(gloria_folder) # change directory from working dir to dir with files

for year in years:
    gloria_log = pymrio.download_gloria(storage_folder=gloria_folder, year=year, overwrite_existing=True)
    
    for item in os.listdir(gloria_folder): # loop through items in dir
        if item.endswith('.zip'): # check for ".zip" extension
            file_name = gloria_folder + '/' + os.path.abspath(item).split('\\')[-1] # get full path of files
            with zipfile.ZipFile(file_name) as file: # create zipfile object temporarily
                file.extractall(gloria_folder) # extract file to dir
                file.close() # close file
            
            os.remove(file_name) # delete zipped file
        
filenames = os.listdir(gloria_folder) # list files so they can be renamed
for filename in filenames:
    if filename.split('.')[-1] == 'csv':
        print(filename)
        new_name = filename.split('-')[0][-1] + '_' + filename.split('-')[1].split('_')[1] + '.csv'
        os.rename(gloria_folder + '/' + filename, gloria_folder + '/' + new_name)



# WIOD

wiod_data = {}



