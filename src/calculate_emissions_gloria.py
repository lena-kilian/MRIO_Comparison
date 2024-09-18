# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:31:51 2023

@author: geolki

updated by Julia Crook (CEMAC) to optimize the code by only reading what is necessary from csv files
and by computing footprint using the components rather than a big matrix

"""

# use pymrio https://pymrio.readthedocs.io/en/latest/notebooks/autodownload.html

# conda activate mrio

import pandas as pd

import calculate_emissions_functions_gloria as cef



#-------------------------------------------------------------------------------------
# This should be run as python calculate_emission_gloria <config> <start_year> <end_year> [-n -v]
# where config is a config file defining pathnames of files to read and output directory
#       start_year and end_year define the years to read
#       -n means run using minimum data (otherwise do it the old way with big matrices)
#       -v = verbose
#-------------------------------------------------------------------------------------

new=True # run in the old way
fextra='_old' # used for me to output files for testing
fout_extra='' # used for the footprint file to indicate we used the new/old way to do this

config_file='O:/ESCoE_Project/data/MRIO/Gloria/config_large.cfg'
start_year=2010
end_year=2010


# read config file to get filenames
mrio_filepath, outdir, lookup_filepath, labels_fname, lookup_fname, Z_fname, Y_fname, co2_fname, gloria_version = cef.read_config(config_file)

z_idx, industry_idx, product_idx, iix,pix,y_cols, sat_rows = cef.get_metadata_indices(mrio_filepath, lookup_filepath, labels_fname, lookup_fname)

stressor_cat = "'GHG_total_EDGAR_consistent'" # use this to extract correct row from stressor dataset below. Only one row from this DF is needed in the analysis
#"'co2_excl_short_cycle_org_c_total_EDGAR_consistent'" 

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

    outfile=outdir+'Gloria_CO2_' + str(year) + fout_extra+'.csv'

    S, U, Y, stressor = cef.read_data_new(z_filepath, y_filepath, co2_filepath, iix, pix, industry_idx, product_idx, y_cols, stressor_row)

    print('Data loaded for ' + str(year))

    footprint = cef.indirect_footprint_SUT_new(S, U, Y, stressor)    

    print('Footprint calculated for ' + str(year))

    footprint.to_csv(outfile)
    print('Footprint saved for ' + str(year))

print('Gloria Done')
