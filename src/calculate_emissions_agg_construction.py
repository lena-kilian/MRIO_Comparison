# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:31:51 2023

@author: geolki
"""

import pandas as pd
from sys import platform
import calculate_emissions_functions as cef
import calculate_emissions_functions_gloria as cef_g
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

years = range(2010, 2021)

version = '2024'
footprint = 'ghg'

#lookup = pd.read_excel(wd + 'ESCoE_Project/data/lookups/mrio_lookup_sectors_countries_finaldemand.xlsx', sheet_name=None) 

exio_lookup = {'Construction (45)':'Construction', 
               'Construction work (45)':'Construction',
               'Re-processing of secondary construction material into aggregates':'Construction',
               'Secondary construction material for treatment, Re-processing of secondary construction material into aggregates':'Construction'}

gloria_lookup = {' Building construction industry':' Construction industry', 
                 ' Civil engineering construction industry':' Construction industry',
                 ' Building construction product':' Construction product', 
                 ' Civil engineering construction product':' Construction product'}

##############
## Exiobase ##
##############

stressor_var = 'GHG emissions AR5 (GWP100) | GWP100 (IPCC, 2010)'
# 'Carbon dioxide (CO2) IPCC categories 1 to 4 and 6 to 7 (excl land use, land use change and forestry)'

co2_exio_all = {}

for year in years:
    
    exio_data = {}
                  
    filepath = wd + 'UKMRIO_Data/EXIOBASE/3.8.2/MRSUT_' + str(year) + '/'
            
    exio_data['S'] = pd.read_csv(filepath + 'supply.csv', sep='\t', header = [0, 1], index_col = [0, 1]).T
    exio_data['U'] = pd.read_csv(filepath + 'use.csv', sep='\t', header = [0, 1], index_col = [0, 1])
    exio_data['Y'] = pd.read_csv(filepath + 'final_demand.csv', sep='\t', header = [0, 1], index_col = [0, 1])
    exio_data['co2'] = pd.DataFrame(pd.read_csv(filepath + 'F.txt', sep='\t', index_col=0, header=[0, 1]).loc[stressor_var, :])
    
    # calculate exio footprint
    S = exio_data['S']; U = exio_data['U']; Y = exio_data['Y']; stressor = exio_data['co2'].iloc[:,0]
    # aggregate construction
    S = S.rename(columns=exio_lookup, index=exio_lookup).sum(axis=0, level=[0,1]).sum(axis=1, level=[0,1])
    U = U.rename(columns=exio_lookup, index=exio_lookup).sum(axis=0, level=[0,1]).sum(axis=1, level=[0,1])
    Y = Y.rename(index=exio_lookup).sum(axis=0, level=[0,1])
    stressor = stressor.rename(index=exio_lookup).sum(axis=0, level=[0,1])
    
    co2_exio = cef.indirect_footprint_SUT_exio(S, U, Y, stressor)
    print(year)
    
    co2_exio.to_csv(emissions_filepath + 'Exiobase/Exiobase_emissions_' + str(year) + '_agg_construction.csv')
    
    co2_exio_all[year] = co2_exio

pickle.dump(co2_exio_all, open(emissions_filepath + 'Exiobase/Exiobase_emissions_agg_construction.p', 'wb'))


############
## Gloria ##
############

config_file= wd + 'ESCoE_Project/data/MRIO/Gloria/config_large.cfg'
start_year=years[0]
end_year=years[-1]

# read config file to get filenames
mrio_filepath, outdir, lookup_filepath, labels_fname, lookup_fname, Z_fname, Y_fname, co2_fname, gloria_version = cef_g.read_config(config_file)

z_idx, industry_idx, product_idx, iix,pix,y_cols, sat_rows = cef_g.get_metadata_indices(mrio_filepath, lookup_filepath, labels_fname, lookup_fname)

# use this to extract correct row from stressor dataset below. Only one row from this DF is needed in the analysis
stressor_cat = "'GHG_total_EDGAR_consistent'" 

# JAC work out which row stressor_cat is on
stressor_row = pd.Index(sat_rows).get_loc(stressor_cat)

# define sample year, normally this is: range(2010, 2019)
# here years is now determined from inputs,
# it used to be a range(2010, 2019). In future work this will likley be range(2001, 2023)
for year in range(start_year,end_year+1):

    # set up filepaths
    # file name changes from 2017, so define this here

    split=Z_fname.split('%')
    if len(split)>1:
        z_filepath=mrio_filepath+gloria_version+split[0]+str(year)+split[1]
    else:
        z_filepath=mrio_filepath+gloria_version+Z_fname

    split=Y_fname.split('%')
    if len(split)>1:
        y_filepath=mrio_filepath+gloria_version+split[0]+str(year)+split[1]
    else:
        y_filepath=mrio_filepath+gloria_version+Y_fname

    split=co2_fname.split('%')
    if len(split)>1:
        co2_filepath=mrio_filepath+gloria_version+'Env_extensions/'+split[0]+str(year)+split[1]
    else:
        co2_filepath=mrio_filepath+gloria_version+'Env_extensions/'+co2_fname

    outfile=outdir+'Gloria_CO2_' + str(year) + '_agg_construction.csv'

    S, U, Y, stressor = cef_g.read_data_new(z_filepath, y_filepath, co2_filepath, iix, pix, industry_idx, product_idx, y_cols, stressor_row)
    # aggregate construction
    S = S.rename(columns=gloria_lookup, index=gloria_lookup).sum(axis=0, level=[0,1]).sum(axis=1, level=[0,1])
    U = U.rename(columns=gloria_lookup, index=gloria_lookup).sum(axis=0, level=[0,1]).sum(axis=1, level=[0,1])
    Y = Y.rename(index=gloria_lookup).sum(axis=0, level=[0,1])
    stressor = stressor.rename(columns=gloria_lookup).sum(axis=1, level=[0,1])

    print('Data loaded for ' + str(year))

    footprint = cef_g.indirect_footprint_SUT_new(S, U, Y, stressor)    

    print('Footprint calculated for ' + str(year))

    footprint.to_csv(outfile)
    print('Footprint saved for ' + str(year))

print('Gloria Done')
    