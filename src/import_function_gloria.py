# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:31:51 2023

@author: geolki

updated by Julia Crook (CEMAC) to optimize the code by only reading what is necessary from csv files
and by computing footprint using the components rather than a big matrix
I have added read_config to get the required parameters to run either on Large data or Small data. 
Also it makes the code more configurable in where the files are stored.

"""

import pandas as pd
import numpy as np
#import scipy.linalg as sla


#----------------------------------------------------------------
# read config file to get indir, outdir and filenames which are different
# for small and large data
#----------------------------------------------------------------
def read_config(cfname):

    cf = open(cfname, "r")
    lines=cf.readlines()
    if len(lines)<7:
        print('invalid config file')
        raise SystemExit
    # need to remove '\n' from line
    indir=lines[0][:-1]
    outdir=lines[1][:-1]
    lookupdir=lines[2][:-1]
    labels_fname=lines[3][:-1]
    lookup_fname=lines[4][:-1]
    Z_fname=lines[5][:-1]
    Y_fname=lines[6][:-1]
    co2_fname=lines[7][:-1]
    gloria_version=lines[8][:-1]
    cf.close()

    return indir, outdir, lookupdir, labels_fname, lookup_fname, Z_fname, Y_fname, co2_fname, gloria_version 


#######################################################################################
## Gloria Metadata
## The next two funtions are for reading the lookup and labels excel files to determine
## the multiIndices for Z and Y
#######################################################################################
#----------------------------------------------------------------
# this function will split the strings into a list of <country>+ (<code>), 
# and <some_str>, where <some_str> is the remainder of the string
# inputs:
#   labels is a list of strings containing <country> (<code>) <some_str>
#       note that the string before the code may also contain ()
#   valid_country_codes is a list of valid country codes
# returns:
#   country - list of <country>
#   country_code - list of <code>
#   remainder - list of <some_str> 
#------------------------------------------------------
def split_country_and_code(labels, valid_country_codes):

    nl=len(labels)
    country = ['NA']*nl
    country_code=['']*nl
    remainder=['']*nl
    n=0
    for cs in labels:
        items=cs.split('(')
        # find which part has the country code
        for item in items:
            items2=item.split(')')
            if items2[0] in valid_country_codes:
                country_code[n]=items2[0]
                # country is the string before the country code and remainder is the part after
                items3=cs.split('('+items2[0]+')')
                country[n]=items3[0]
                remainder[n]=items3[1]
        if country[n]=='NA':
            print('Could not find country code in', cs)
        n+=1


    # end code if there is a mismatch in labels, this is done to test that the code does what it should
    if 'NA' in country:
        print('Error: Missing country labels')
        raise SystemExit 
        
    return country, country_code, remainder


#----------------------------------------------------------------
# This function reads the excel files containing the indices in Z
# mrio_filepath is the directory where they exist
# labels_fname is the filename for the labels file
# lookup_fname is the filename for the lookup file
#----------------------------------------------------------------

def get_metadata_indices(mrio_filepath, lookup_filepath, labels_fname, lookup_fname):
    # read metadata which is used for column and row labels later on
    readme = mrio_filepath + labels_fname
    labels = pd.read_excel(readme, sheet_name=None)

    # get lookup to fix labels
    lookup = pd.read_excel(lookup_filepath + lookup_fname, sheet_name=None)
    # get list of countries in dataset
    lookup['countries'] = lookup['countries'][['gloria', 'gloria_code']].drop_duplicates().dropna()
    lookup['countries']['gloria_combo'] = lookup['countries']['gloria'] + ' (' + lookup['countries']['gloria_code'] + ') '
    # get list of sectors in dataset
    lookup['sectors'] = lookup['sectors'][['gloria']].drop_duplicates().dropna()

    # fix Z labels
    # This helps beng able to split the 'country' from the 'sector' component in the label later on
    # Essentially this removes some special characters from the country names, making it easier to split the country from the secro by a specific charater later on
    t_cats = pd.DataFrame(labels['Sequential region-sector labels']['Sequential_regionSector_labels']).drop_duplicates(); t_cats.columns = ['label']
    # remove special characters frm country names -JAC this seems to be only removing sector
    # JAC replace loop with call to function
    valid_country_codes=lookup['countries']['gloria_code'].tolist()
    country, country_code, remainder=split_country_and_code(t_cats['label'], valid_country_codes)        
    #t_cats['country'] = country # dont really need this as not used
    t_cats['country_code'] = country_code
    t_cats['sector'] = remainder

    # fix final demand labels (this is in the Y dataframe later)
    # it follows the same logic as the Z label fix above
    fd_cats = pd.DataFrame(labels['Sequential region-sector labels']['Sequential_finalDemand_labels'].dropna(how='all', axis=0)); fd_cats.columns = ['label']
    # remove special characters frm country names
    # JAC replace loop below with call to function
    country, country_code, remainder=split_country_and_code(fd_cats['label'], valid_country_codes)        
    #fd_cats['country'] = country # dont really need this as not used
    fd_cats['country_code'] = country_code
    fd_cats['fd'] = remainder

    # split lables by industries vs products (these later get split into the S and U dataframes)
    t_cats['ind'] = t_cats['label'].str[-8:]
    industries = t_cats.loc[t_cats['ind'] == 'industry']
    products = t_cats.loc[t_cats['ind'] == ' product']
    sector_type=np.asarray(t_cats['ind'])
    iix=np.where(sector_type=='industry')[0]
    pix=np.where(sector_type==' product')[0]

    # make index labels
    z_idx = pd.MultiIndex.from_arrays([t_cats['country_code'], t_cats['sector']]) # labels for Z dataframe
    industry_idx = pd.MultiIndex.from_arrays([industries['country_code'], industries['sector']]) # labels used to split Z into S and U
    product_idx = pd.MultiIndex.from_arrays([products['country_code'], products['sector']]) # labels used to split Z into S and U
    y_cols = pd.MultiIndex.from_arrays([fd_cats['country_code'], fd_cats['fd']]) # labels for Y dataframe

    sat_rows = labels['Satellites']['Sat_indicator'] # labels for CO2 dataframe
    # clear space in variable explorer to free up memory
    # del t_cats, temp_c, cs, a, c, temp_s, i, temp, fd_cats, industries, products, item
    
    # running the code to this point normally takes around 15 seconds
    return z_idx, industry_idx, product_idx, iix,pix, y_cols, sat_rows

###########################################################################################
## Reading Data
## I now have 2 versions of the code that reads the data and processes into the footprint
## The old version that uses big matrices and a new version that just uses the components
## that are actually useful, ie non zero
## the read_data_new/old do the following:
##     reads S and U from Z (z_filepath)
##     reads the product rows from Y (y_filepath)
##     read the industry columns and stressor_row from co2_filepath.
##
## In read_data_new, only load required rows and cols from everything
## There are 2 version of cef.indirect_footprint_SUT to calcluate the footprint
## cef.indirect_footprint_SUT_new calculates the footprint using only the data given
## without constructing big matrices again
###########################################################################################

#--------------------------------------------------------------------
# function read_data_new
# This reads only the relevant rows and columns from Z, Y and co2file
# this is the new way to get the data using much less memory
# inputs:
#     z_filepath - the filepath from where to read the big matrix Z
#     y_filepath - the filepath from where to read Y
#     co2_filepath - the filepath from where to read the stressor data
#     iix -  these are the indices (0,1....) where the industry rows/columns are
#     pix - these are the indices (0,1...) where the product rows/columns are
#     industry_idx - this is the multiIndex to set in S/U/stressor
#     product_idx -  this is the multiIndex to set in S/U/Y
#     y_cols - the multiIndex to set in Y columns
#     stressor_row - the row to pick out from co2_filepath
# returns:
#     S - the [industry_idx, product_idx] part of Z
#     U - the [product_idx, industry_idx] part of Z
#     Y - the product_idx rows of Y
#     stressor - the [stressor_row, industry_idx] part of co2_filepath (NB this is a single row)
#--------------------------------------------------------------------

def read_data_new(z_filepath, y_filepath, iix, pix, industry_idx, product_idx, y_cols):
    # read S and U directly from Z by specifying specific rows and cols
    S=pd.read_csv(z_filepath, header=None, index_col=None, skiprows = lambda x: x not in iix, usecols=pix)
    S.index=industry_idx; S.columns=product_idx
    #U=pd.read_csv(z_filepath, header=None, index_col=None,skiprows = lambda x: x not in pix, usecols=iix)
    #U.index=product_idx; U.columns=industry_idx

    # read product rows of Y and rename index and column
    # JAC just read the required rows directly and set up the indices for the columns
    Y = pd.read_csv(y_filepath, header=None, index_col=None, skiprows = lambda x: x not in pix)
    Y.index=product_idx; Y.columns=y_cols
    
    # import stressor (co2) data
    # JAC just read the required row and columns directly from the csv
    #stressor = pd.read_csv(co2_filepath, header=None, index_col=None, nrows=1, skiprows=stressor_row, usecols=iix)
    #stressor.columns=industry_idx

    return S, Y#, U, stressor
