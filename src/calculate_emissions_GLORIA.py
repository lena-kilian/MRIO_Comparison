# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:31:51 2023

@author: geolki
"""

import pandas as pd
from sys import platform
import calculate_emissions_functions_gloria as cef_g


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
years = range(2005, 2020)
    

############
## Gloria ##
############

lookup = pd.read_excel('O:/ESCoE_Project/data/MRIO/Gloria/GLORIA_ReadMe_059a.xlsx', sheet_name='Regions')
country_dict = dict(zip(lookup['Region_acronyms'], lookup['Region_names']))

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

    split=co2_fname.split('%')
    if len(split)>1:
        co2_filepath=gloria_filepath+gloria_version+'Env_extensions/'+split[0]+str(year)+split[1]
    else:
        co2_filepath=gloria_filepath+gloria_version+'Env_extensions/'+co2_fname

    S, U, Y, stressor = cef_g.read_data_new(z_filepath, y_filepath, co2_filepath, iix, pix, industry_idx, product_idx, y_cols, stressor_row)

    # save for reference
    emissions_ind = cef_g.indirect_footprint_SUT_ind(S, U, Y, stressor).T
    
    emissions_ind_gov = emissions_ind.swaplevel(axis=1)[[' Government final consumption P.3g']].swaplevel(axis=1)
    emissions_ind_gov = emissions_ind_gov.rename(columns=country_dict, index=country_dict)

    # save as csv
    emissions_ind_gov.to_csv('C:/Users/geolki/OneDrive - University of Leeds/Postdoc/Gloria_detail/Gloria_' + footprint + '_industries_' + str(year) + '_governments.csv')
    
    print(year)