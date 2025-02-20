# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:31:51 2023

@author: geolki
"""

import pandas as pd
from sys import platform
import import_function_gloria as imp_g
import numpy as np

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

lookup_cat =  'abc' #'combined_name' #  # 
row_lookup = pd.read_excel('O:/ESCoE_Project/data/lookups/GLORIA_small_sectors.xlsx', sheet_name='Countries')[['gloria_code', lookup_cat]].drop_duplicates()
row_dict = dict(zip(row_lookup['gloria_code'], row_lookup[lookup_cat]))

# read config file to get filenames
config_file= wd + 'ESCoE_Project/data/MRIO/Gloria/config_large.cfg'
gloria_filepath, outdir, lookup_filepath, labels_fname, lookup_fname, Z_fname, Y_fname, co2_fname, gloria_version = imp_g.read_config(config_file)

z_idx, industry_idx, product_idx, iix, pix, y_cols, sat_rows = imp_g.get_metadata_indices(gloria_filepath, lookup_filepath, labels_fname, lookup_fname)

# define sample year, normally this is: range(2010, 2019)
# here years is now determined from inputs,
# it used to be a range(2010, 2019). In future work this will likley be range(2001, 2023)
for year in years:
    print('start', year)

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
        
    S = {}; Y = {}
    for i in range(1, 6):
        S[i], Y[i] = imp_g.read_data_new(z_filepath.replace('Markup001', 'Markup00' + str(i)), y_filepath.replace('Markup001', 'Markup00'  + str(i)), iix, pix, industry_idx, product_idx, y_cols)
        print(i)
    #S = S[1]# + S[4] - S[5] + S[2] + S[3]
    Y = Y[1] + Y[4] - Y[5] + Y[2] + Y[3]
    
    Y.to_csv('C:/Users/geolki/OneDrive - University of Leeds/Postdoc/Gloria_detail/Spend/FD_Gloria_industry_' + str(year) + '_pp_Y_raw.csv') #_purchaseprices
        
    # 1 = basic prices, 2 = trade margins, 3 = transport margins, 4 = taxes on products, 5 = subsidies on products
    # Purchaser prices = Basic prices + taxes on products (excluding VAT) - subsidies on products + trade and transport margins + non-deductible VAT
    # Purchaser price = S[1] + S[4] - S[5] + S[2] + S[3]
    '''
    # aggregate countries
    S = S.rename(index=row_dict, columns=row_dict).sum(axis=0, level=[0, 1]).sum(axis=1, level=[0, 1])
    Y = Y.rename(index=row_dict, columns=row_dict).sum(axis=0, level=[0, 1]).sum(axis=1, level=[0, 1])
    
    # convert to industry by industry table
    # Industry-by-industry input-output table based on fixed product sales structure assumption 

    q = U.sum(1) + Y.sum(1) # need to rerun with U and Y
    #q = q.loc[S_new.columns]
    q_diag = np.diag(q)
    q_diag_inv = np.linalg.inv(q_diag)
     
    T = np.dot(np.array(S), q_diag_inv)

    
    # Industry-by-industry input-output table based on fixed industry sales structure assumption 
    g = S.sum(1)
    #q = q.loc[S_new.columns]
    g_diag = np.diag(g)
    inv_Vt = np.linalg.inv(S.T)
     
    T = np.dot(g_diag, inv_Vt)
    
    F = pd.DataFrame(index=S.index)
    for item in Y.columns.tolist():
        Y_small = Y.loc[S.columns, item]
        temp = np.dot(T, Y_small)
        F[item] = temp
    
    # save final demand
    F.to_csv('C:/Users/geolki/OneDrive - University of Leeds/Postdoc/Gloria_detail/Spend/FD_Gloria_industry_' + str(year) + '_Eurostat_method_v2.csv') #_purchaseprices
    '''
    print('end', year)