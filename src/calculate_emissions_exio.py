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
import pickle 

# set working directory
# make different path depending on operating system
if platform[:3] == 'win':
    wd = 'O://'
else:
    wd = r'/Volumes/a72/' 

# define filepaths
mrio_filepath = wd + 'ESCoE_Project/data/MRIO/'
emissions_filepath = wd + 'ESCoE_Project/data/Emissions/'
outputs_filepath = wd + 'UKMRIO_Data/outputs/results_2023/'

stressor_var = 'GHG emissions AR5 (GWP100) | GWP100 (IPCC, 2010)'
# 'Carbon dioxide (CO2) IPCC categories 1 to 4 and 6 to 7 (excl land use, land use change and forestry)'

years = range(2010, 2021)

##############
## EXIOBASE ##
##############

results = {}

for year in years:
    
    exio_data = {}
                  
    filepath = wd + 'UKMRIO_Data/EXIOBASE/3.8.2/MRSUT_' + str(year) + '/'
            
    exio_data['S'] = pd.read_csv(filepath + 'supply.csv', sep='\t', header = [0, 1], index_col = [0, 1]).T
    exio_data['U'] = pd.read_csv(filepath + 'use.csv', sep='\t', header = [0, 1], index_col = [0, 1])
    exio_data['Y'] = pd.read_csv(filepath + 'final_demand.csv', sep='\t', header = [0, 1], index_col = [0, 1])
    exio_data['co2'] = pd.DataFrame(pd.read_csv(filepath + 'F.txt', sep='\t', index_col=0, header=[0, 1]).loc[stressor_var, :])
    
    # calculate exio footprint

    S = exio_data['S']; U = exio_data['U']; Y = exio_data['Y']; stressor = exio_data['co2'].iloc[:,0]
    co2_exio = cef.indirect_footprint_SUT_exio(S, U, Y, stressor)
    print(year)
    
    co2_exio.to_csv(emissions_filepath + 'Exiobase/Exiobase_emissions_' + str(year) + '.csv')
    print(year, co2_exio.loc['GB', 'GB'].sum().sum())
    
    results[year] = co2_exio

pickle.dump(results, open(emissions_filepath + 'Exiobase/Exiobase_emissions.p', 'wb'))