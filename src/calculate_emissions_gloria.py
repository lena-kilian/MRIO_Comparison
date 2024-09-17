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
from sys import argv
import numpy as np


from calculate_emissions_functions import *



#-------------------------------------------------------------------------------------
# This should be run as python calculate_emission_gloria <config> <start_year> <end_year> [-n -v]
# where config is a config file defining pathnames of files to read and output directory
#       start_year and end_year define the years to read
#       -n means run using minimum data (otherwise do it the old way with big matrices)
#       -v = verbose
#-------------------------------------------------------------------------------------
def main():

    new=False # run in the old way
    fextra='_old' # used for me to output files for testing
    fout_extra='' # used for the footprint file to indicate we used the new/old way to do this
    verbose=False
    # in the old code it calculated e.L but should do L.e use this flag to define whether we should make the correction
    if len(argv)<4:
        print('Useage: python', argv[0], '<config> <start_year> <end_year> [-n -v]')
        print('where -n means use new way to process data,\n-v means verbose')
        exit()

    config_file=argv[1]
    start_year=int(argv[2])
    end_year=int(argv[3])
    for i in range(4,len(argv)):
        if argv[i]=='-n':
            new=True
            print('NEW')
            fextra='_new'
            fout_extra='_New'
        elif argv[i]=='-v':
            verbose=True


    # read config file to get filenames
    mrio_filepath, outdir, labels_fname, lookup_fname, Z_fname, Y_fname, co2_fname = read_config(config_file)

    z_idx, industry_idx, product_idx, iix,pix,y_cols, sat_rows=get_metadata_indices(mrio_filepath,labels_fname, lookup_fname)

    stressor_cat = "'co2_excl_short_cycle_org_c_total_EDGAR_consistent'" # use this to extract correct row from stressor dataset below. Only one row from this DF is needed in the analysis

    if new:
        # JAC work out which row stressor_cat is on
        stressor_row = pd.Index(sat_rows).get_loc(stressor_cat)

    # define sample year, normally this is: range(2010, 2019)
    # here years is now determined from inputs,
    # it used to be a range(2010, 2019). In future work this will likley be range(2001, 2023)
    for year in range(start_year,end_year+1):

        # set up filepaths
        # file name changes from 2017, so define this here
        if year < 2017:
            date_var = '20230314'
        else:
            date_var = '20230315'

        split=Z_fname.split('%')
        if len(split)>1:
            z_filepath=mrio_filepath+split[0]+date_var+split[1]+str(year)+split[2]
        else:
            z_filepath=mrio_filepath+Z_fname

        split=Y_fname.split('%')
        if len(split)>1:
            y_filepath=mrio_filepath+split[0]+date_var+split[1]+str(year)+split[2]
        else:
            y_filepath=mrio_filepath+Y_fname

        split=co2_fname.split('%')
        if len(split)>1:
            co2_filepath=mrio_filepath+split[0]+str(year)+split[1]
        else:
            co2_filepath=mrio_filepath+co2_fname

        outfile=outdir+'Gloria_CO2_' + str(year) + fout_extra+'.csv'

        if new:
            S, U, Y, stressor=read_data_new(z_filepath, y_filepath, co2_filepath, iix, pix, industry_idx, product_idx, y_cols, stressor_row)


        else:    
            S, U, Y, stressor=read_data_old(z_filepath, y_filepath, co2_filepath, z_idx,industry_idx, product_idx, y_cols, sat_rows, stressor_cat)


        if verbose:
            print('DBG: size S, U', S.shape, U.shape)
            print('DBG: size Y', Y.shape, Y.to_numpy().shape)
            print('DBG: size stressor', stressor.shape)
            #np.save('Y_'+fextra+'.npy', Y.to_numpy())

            print('Data loaded for ' + str(year))

        if new:    
            footprint=indirect_footprint_SUT_new(S, U, Y, stressor,verbose)    
        else:
            footprint=indirect_footprint_SUT(S, U, Y, stressor, verbose)    


        if verbose:
            print('Footprint calculated for ' + str(year))
    
        footprint.to_csv(outfile)
        print('Footprint saved for ' + str(year))

    print('Gloria Done')


if __name__ == '__main__':
    main()
