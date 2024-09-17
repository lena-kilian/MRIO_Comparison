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
import numpy as np


# set working directory
# make different path depending on operating system
if platform[:3] == 'win':
    wd = 'O://'
else:
    wd = r'/Volumes/a72/' 

# define filepaths
mrio_filepath = wd + 'ESCoE_Project/data/MRIO/'

# define sample year, normally this is: range(2010, 2019)
years = [2016]

#####################
## Gloria Metadata ##
#####################

# read metadata which is used for column and row labels later on
readme = mrio_filepath + 'Gloria/GLORIA_ReadMe_057.xlsx'
labels = pd.read_excel(readme, sheet_name=None)

# get lookup to fix labels
lookup = pd.read_excel('O://ESCoE_Project/data/lookups/mrio_lookup_sectors_countries_finaldemand.xlsx', sheet_name=None)
# get list of countries in dataset
lookup['countries'] = lookup['countries'][['gloria', 'gloria_code_long']].drop_duplicates().dropna()
lookup['countries']['gloria_combo'] = lookup['countries']['gloria'] + ' (' + lookup['countries']['gloria_code_long'] + ') '
# get list of sectors in dataset
lookup['sectors'] = lookup['sectors'][['gloria']].drop_duplicates().dropna()

# fix Z labels
# This helps beng able to split the 'country' from the 'sector' component in the label later on
# Essentially this removes some special characters from the country names, making it easier to split the country from the secro by a specific charater later on
t_cats = pd.DataFrame(labels['Sequential region-sector labels']['Sequential_regionSector_labels']).drop_duplicates(); t_cats.columns = ['label']
# remove special characters frm country names
temp_c = []
for cs in t_cats['label']:
    a = False
    for item in cs.split('('):
        if item.split(')')[0] in lookup['countries']['gloria_code_long'].tolist():
            a = True
            c = cs.split(item.split(')')[0])[0] + item.split(')')[0] + ')'
            temp_c.append(c)
    if a == False:
        temp_c.append('NA')

# end code if there is a mismatch in labels, this is done to test that the code does what it should
if 'NA' in temp_c:
    print('Missing coutry labels')
    raise SystemExit 
        
t_cats['country_full'] = temp_c
t_cats['country'] = [x.split('(')[-1][:-1] for x in t_cats['country_full']]
temp_s = []
for i in range(len(t_cats)):
    temp = t_cats.iloc[i, :]
    temp_s.append(temp['label'].replace(temp['country_full'], ''))
t_cats['sector'] = temp_s

# fix final demand labels (this is in the Y dataframe later)
# it follows the same logic as the Z label fix above
fd_cats = pd.DataFrame(labels['Sequential region-sector labels']['Sequential_finalDemand_labels'].dropna(how='all', axis=0)); fd_cats.columns = ['label']
# remove special characters frm country names
temp_c = []
for cs in fd_cats['label']:
    a = False
    for item in cs.split('('):
        if item.split(')')[0] in lookup['countries']['gloria_code_long'].tolist():
            a = True
            c = cs.split(item.split(')')[0])[0] + item.split(')')[0] + ')'
            temp_c.append(c)
    if a == False:
        temp_c.append('NA')
   
# end code if there is a mismatch in labels, this is done to test that the code does what it should
if 'NA' in temp_c:
    print('Missing coutry labels')
    raise SystemExit

fd_cats['country_full'] = temp_c
fd_cats['country'] = [x.split('(')[-1][:-1] for x in fd_cats['country_full']]
temp_s = []
for i in range(len(fd_cats)):
    temp = fd_cats.iloc[i, :]
    temp_s.append(temp['label'].replace(temp['country_full'], ''))
fd_cats['fd'] = temp_s

# split lables by ndustries vs products (these later get split into the S and U dataframes)
t_cats['ind'] = t_cats['label'].str[-8:]
industries = t_cats.loc[t_cats['ind'] == 'industry']
products = t_cats.loc[t_cats['ind'] != 'industry']

# make index labels
z_idx = pd.MultiIndex.from_arrays([t_cats['country'], t_cats['sector']]) # labels for Z dataframe
industry_idx = pd.MultiIndex.from_arrays([industries['country'], industries['sector']]) # labels used to split Z into S and U
product_idx = pd.MultiIndex.from_arrays([products['country'], products['sector']]) # labels used to split Z into S and U
y_cols = pd.MultiIndex.from_arrays([fd_cats['country'], fd_cats['fd']]) # labels for Y dataframe

sat_rows = labels['Satellites']['Sat_indicator'] # labels for CO2 dataframe
stressor_cat = "'co2_excl_short_cycle_org_c_total_EDGAR_consistent'" # use this to extract correct row from stressor dataset below. Only one row from this DF is needed in the analysis

# clear space in variable explorer to free up memory
del t_cats, temp_c, cs, a, c, temp_s, i, temp, fd_cats, industries, products, item
    
# runnng the code to this point normally takes around 15 seconds

######################
## Gloria Emissions ##
######################

for year in years: # here years is only [2016], normally this is range(2010, 2019). In future work this will likley be range(2001, 2023)
    # file name ending changes from 2017, so define this here
    if year < 2017:
        date_var = '20230314'
    else:
        date_var = '20230315'
        
    # define filenames to make script cleaner
    z_filepath = (mrio_filepath + 'Gloria/Main/' + date_var + '_120secMother_AllCountries_002_T-Results_' + str(year) + '_057_Markup001(full).csv') 
    y_filepath = (mrio_filepath + 'Gloria/Main/' + date_var + '_120secMother_AllCountries_002_Y-Results_' + str(year) + '_057_Markup001(full).csv') 
    co2_filepath = (mrio_filepath + 'Gloria/Satellite_Accounts/20230727_120secMother_AllCountries_002_TQ-Results_' + str(year) + '_057_Markup001(full).csv') 
    
    # import Z file to make S and U tables
    # mathematically this is probably not necessary, but it then follows the same structure as the other datasets we use, which is why this is currenlty done
    # Mainly this just allows us to use a single function for multiple datasets
    # S and U are later combined into Z again, but with  slightly different index and column order
    Z = pd.read_csv(z_filepath, header=None, index_col=None)
    Z.index = z_idx; Z.columns = z_idx
    S = Z.loc[industry_idx, product_idx]
    U = Z.loc[product_idx, industry_idx]
    del Z # remove Z to clear memory
    
    # import Y and rename index and column
    # again, it's matched to the strutcure of other datasets we analyse
    Y = pd.read_csv(y_filepath, header=None, index_col=None)
    Y.index = z_idx; Y.columns = y_cols    
    Y = Y.loc[product_idx]
    
    # import stressor (co2) data
    # again, it's matched to the strutcure of other datasets we analyse
    stressor = pd.read_csv(co2_filepath, header=None, index_col=None)
    stressor.index = sat_rows; stressor.columns = z_idx
    stressor = stressor.loc[stressor_cat, industry_idx]
    
    print('Data loaded for ' + str(year))
    
    #### The part from here to line 191 normally sits in a function - I took it out to see which part is slow but it is 'cef.indirect_footprint_SUT'
    
    # save column names - so that 
    su_idx = pd.MultiIndex.from_arrays([[x[0] for x in S.columns.tolist()] + [x[0] for x in U.columns.tolist()],
                                        [x[1] for x in S.columns.tolist()] + [x[1] for x in U.columns.tolist()]])
    u_cols = U.columns.tolist()

    # calculate gloria footprint
    Z = cef.make_Z_from_S_U(S, U) 
    del S, U # remove S and U to clear memory
    bigY = np.zeros(shape = [np.size(Y, 0)*2, np.size(Y, 1)])
    
    footprint = np.zeros(shape = bigY.shape).T
    
    bigY[np.size(Y, 0):np.size(Y, 0)*2, 0:] = Y     
    x = cef.make_x(Z, bigY)
    L = cef.make_L(Z, x)
    bigstressor = np.zeros(shape = [np.size(Y, 0)*2, 1])
    bigstressor[:np.size(Y, 0), 0] = np.array(stressor)
    e = np.sum(bigstressor, 1)/x
    eL = np.dot(e, L)
    
    for a in range(1): # normally this is range(np.size(Y, 1)):
        footprint[a] = np.dot(eL, np.diag(bigY[:, a]))
    
    footprint = pd.DataFrame(footprint, index=Y.columns, columns=su_idx)
    footprint = footprint[u_cols]
    
    ######### function ends here
    
    print('Footprint calculated for ' + str(year))
    
    footprint.to_csv(wd + 'ESCoE_Project/data/Emissions/Gloria/MEMORY_TEST_CO2_' + str(year) + '.csv')
    
    print('Footprint saved for ' + str(year))

print('Gloria done')