# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:31:51 2023

@author: geolki
"""

# use pymrio https://pymrio.readthedocs.io/en/latest/notebooks/autodownload.html

# conda install -c conda-forge pymrio

import pymrio
import pandas as pd
import numpy as np
from sys import platform


# set working directory
# make different path depending on operating system
if platform[:3] == 'win':
    wd = 'O://'
else:
    wd = r'/Volumes/a72/' 

# define filepaths
data_filepath = wd + 'UKMRIO_Data/data/'
mrio_data_path = wd + 'geolki/data/raw/MRIOs/'

db = ['Exiobase3', 'OECD', 'Figaro', 'Gloria', 'WIOD', 'EORA']

years = range(2010, 2019)

currency_usd_eur = pd.read_excel(wd + 'UKMRIO_Data/ICIO/ReadMe_ICIO2021_CSV.xlsx',  sheet_name='NCU-USD', index_col=[1], header=0).loc['AUT', :][years].astype(float) # conversion rates from ICIO table

##############
## EXIOBASE ##
##############

exio_data = {}

for year in years:
    
    exio_data[year] = {}
                  
    filepath = wd + "UKMRIO_Data/EXIOBASE/3.8.2/MRSUT_{}/".format(str(year))
            
    exio_data[year]['S'] = pd.read_csv(filepath + 'supply.csv', sep='\t', header = [0,1], index_col = [0,1])
    exio_data[year]['U'] = pd.read_csv(filepath + 'use.csv', sep='\t', header = [0,1], index_col = [0,1])
    exio_data[year]['Y'] = pd.read_csv(filepath + 'final_demand.csv', sep='\t', header = [0,1], index_col = [0,1])
    exio_data[year]['v'] = pd.read_csv(filepath + 'value_added.csv', sep='\t', header = [0,1], index_col = 0)
    exio_data[year]['v'] = pd.DataFrame(exio_data[year]['v'].iloc[0:12,:].sum(0))


############
## FIGARO ##
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
    figaro_data[year]['v'] = use_temp.loc[['W2_D1', 'W2_B2A3G', 'W2_D29X39'], figaro_data[year]['S'].columns].sum(axis=0)

    figaro_data[year]['Y'] = use_temp.loc[figaro_data[year]['S'].index, :].drop(figaro_data[year]['S'].columns.tolist(), axis=1)


##########
## OECD ##
##########

oecd_data = {}

for year in years:
    
    oecd_data[year] = {}

    name = wd + 'UKMRIO_Data/ICIO/ICIO2021_' + str(year) + '.csv'         
    icio = pd.read_csv(name, index_col=0)

    cut_off = icio.columns.tolist().index('AUS_HFCE')
    oecd_data[year]['Z'] = icio.iloc[:cut_off, :cut_off]
    oecd_data[year]['Y'] = icio.iloc[:cut_off, cut_off:]
    oecd_data[year]['v'] = icio.loc['VALU':'VALU', :].iloc[:, :cut_off]
    
    
############
## GLORIA ##
############

gloria_data = {}

##########
## EORA ##
##########

eora_data = {}


##########
## WIOD ##
##########

wiod_data = {}



