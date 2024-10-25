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
import copy as cp


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

lookup = pd.read_excel(wd + 'ESCoE_Project/data/lookups/mrio_lookup_sectors_countries_finaldemand.xlsx', sheet_name=None) 

##############
## Exiobase ##
##############

stressor_var = 'GHG emissions AR5 (GWP100) | GWP100 (IPCC, 2010)'
# 'Carbon dioxide (CO2) IPCC categories 1 to 4 and 6 to 7 (excl land use, land use change and forestry)'

# make lookup
lookup_country = lookup['countries'][['exio_code', 'combined_name']].dropna(how='any').drop_duplicates()
lookup_country = dict(zip(lookup_country['exio_code'], lookup_country['combined_name']))

lookup_sectors = lookup['sectors'][['exio', 'combined_name']].dropna(how='any').drop_duplicates()
lookup_sectors = dict(zip(lookup_sectors['exio'], lookup_sectors['combined_name']))

lookup_fd = lookup['final_demand'][['exio', 'combined_name']].dropna(how='any').drop_duplicates()
lookup_fd = dict(zip(lookup_fd['exio'], lookup_fd['combined_name']))


co2_exio = {}
for year in years:
    
    exio_data = {}
                  
    filepath = wd + 'UKMRIO_Data/EXIOBASE/3.8.2/MRSUT_' + str(year) + '/'
            
    exio_data['S'] = pd.read_csv(filepath + 'supply.csv', sep='\t', header = [0, 1], index_col = [0, 1]).T
    exio_data['U'] = pd.read_csv(filepath + 'use.csv', sep='\t', header = [0, 1], index_col = [0, 1])
    exio_data['Y'] = pd.read_csv(filepath + 'final_demand.csv', sep='\t', header = [0, 1], index_col = [0, 1])
    exio_data['co2'] = pd.DataFrame(pd.read_csv(filepath + 'F.txt', sep='\t', index_col=0, header=[0, 1]).loc[stressor_var, :])
    
    # calculate exio footprint

    S = exio_data['S']; U = exio_data['U']; Y = exio_data['Y']; stressor = exio_data['co2'].iloc[:,0]
    
    # aggregate industries and products
    S = S.rename(columns=lookup_country, index=lookup_country).rename(columns=lookup_sectors, index=lookup_sectors).sum(axis=0, level=[0,1]).sum(axis=1, level=[0,1])
    U = U.rename(columns=lookup_country, index=lookup_country).rename(columns=lookup_sectors, index=lookup_sectors).sum(axis=0, level=[0,1]).sum(axis=1, level=[0,1])
    Y = Y.rename(columns=lookup_country, index=lookup_country).rename(columns=lookup_fd, index=lookup_sectors).sum(axis=0, level=[0,1]).sum(axis=1, level=[0,1])
    stressor = stressor.rename(index=lookup_country).rename(index=lookup_sectors).sum(axis=0, level=[0,1])
    
    co2_exio[year] = cef.indirect_footprint_SUT_exio(S, U, Y, stressor).T.sum(axis=0, level=[0, 1]) / 1000000
    #print(year)

pickle.dump(co2_exio, open(emissions_filepath + 'Exiobase/Exiobase_emissions_' + footprint + '_v' + version + '_agg_all.p', 'wb'))

print('Exiobase Done')

############
## Figaro ##
############

co2_figaro = {}

# make lookup
lookup_country = lookup['countries'][['figaro_code', 'combined_name']].dropna(how='any').drop_duplicates()
lookup_country = dict(zip(lookup_country['figaro_code'], lookup_country['combined_name']))

lookup_sectors = lookup['sectors'][['figaro_code', 'combined_name']].dropna(how='any').drop_duplicates()
temp = cp.copy(lookup_sectors)
temp['figaro_code'] = 'CPA_' + temp['figaro_code']
lookup_sectors = lookup_sectors.append(temp)
lookup_sectors = dict(zip(lookup_sectors['figaro_code'], lookup_sectors['combined_name']))

lookup_fd = lookup['final_demand'][['figaro_code', 'combined_name']].dropna(how='any').drop_duplicates()
lookup_fd = dict(zip(lookup_fd['figaro_code'], lookup_fd['combined_name']))


# get figaro footprint
for year in years:

    S = pd.read_csv(mrio_filepath + 'Figaro/Ed_2024/matrix_eu-ic-supply_24ed_' + str(year) + '.csv', index_col=0).T
    S.columns = pd.MultiIndex.from_arrays([[x.split('_')[0] for x in S.columns],  ['_'.join(x.split('_')[1:]) for x in S.columns]])
    S.index = pd.MultiIndex.from_arrays([[x.split('_')[0] for x in S.index], ['_'.join(x.split('_')[1:]) for x in S.index]])
    
    U_all = pd.read_csv(mrio_filepath + 'Figaro/Ed_2024/matrix_eu-ic-use_24ed_' + str(year) + '.csv', index_col=0)
    U_all.columns = pd.MultiIndex.from_arrays([[x.split('_')[0] for x in U_all.columns], ['_'.join(x.split('_')[1:]) for x in U_all.columns]])
    U_all.index = pd.MultiIndex.from_arrays([[x.split('_')[0] for x in U_all.index], ['_'.join(x.split('_')[1:]) for x in U_all.index]])
    
    U = U_all.loc[S.columns, S.index]
 
    y_cols = []
    for item in list(lookup_fd.keys()):
        if item in U_all.columns.levels[1]:
            y_cols.append(item)
            
    Y = U_all.swaplevel(axis=1)[y_cols].swaplevel(axis=1).loc[S.columns,:]
    
    stressor = pd.read_csv(mrio_filepath + 'Figaro/Ed_2024/Env_extensions/' + footprint + 'Footprint_23ed_' + str(year) + '.csv')
    stressor = stressor[['ref_area', 'industry', 'obs_value']].dropna(how='all').drop_duplicates().set_index(['ref_area', 'industry']).loc[S.index].T.sum(axis=1, level=[0,1])
    
    # aggregate industries and products
    S = S.rename(columns=lookup_country, index=lookup_country).rename(columns=lookup_sectors, index=lookup_sectors).sum(axis=0, level=[0,1]).sum(axis=1, level=[0,1])
    U = U.rename(columns=lookup_country, index=lookup_country).rename(columns=lookup_sectors, index=lookup_sectors).sum(axis=0, level=[0,1]).sum(axis=1, level=[0,1])
    Y = Y.rename(columns=lookup_country, index=lookup_country).rename(columns=lookup_fd, index=lookup_sectors).sum(axis=0, level=[0,1]).sum(axis=1, level=[0,1])
    stressor = stressor.rename(columns=lookup_country).rename(columns=lookup_sectors).sum(axis=1, level=[0,1])
    
    # calculate footprint
    co2_figaro[year] = cef.indirect_footprint_SUT_exio(S, U, Y, stressor).T.sum(axis=0, level=[0, 1])

pickle.dump(co2_figaro, open(emissions_filepath + 'Figaro/Figaro_emissions_' + footprint + '_v' + version + '_agg_all.p', 'wb'))

print('Figaro Done')

##########
## OECD ##
##########

oecd_data = {}

oecd_lookup = pd.read_excel(mrio_filepath + 'ICIO/Ed_2024/ReadMe_ICIO_small.xlsx', sheet_name = 'Area_Activities', header=2)
oecd_lookup_country = oecd_lookup[['Code', 'countries']].dropna(how='all')\
    .append(oecd_lookup[['Column1', 'countries']].dropna(how='any').rename(columns={'Column1':'Code'}))
oecd_lookup_industry = oecd_lookup[['Code.1', 'Industry']].dropna(how='all')
oecd_lookup_fd = ['HFCE', 'NPISH', 'GGFC', 'GFCF', 'INVNT', 'DPABR']

oecd_lookup_cols = pd.read_excel(mrio_filepath + 'ICIO/Ed_2024/ReadMe_ICIO_small.xlsx', sheet_name = 'ColItems', header=3, index_col='ID')
oecd_lookup_rows = pd.read_excel(mrio_filepath + 'ICIO/Ed_2024/ReadMe_ICIO_small.xlsx', sheet_name = 'RowItems', header=3, index_col='ID')

temp = pd.read_csv(mrio_filepath + 'ICIO/Ed_2024/Env_extensions/DF_MAIN.csv')
temp = temp.loc[(temp['MEASURE'] == 'PROD_GHG') & (temp['UNIT_MEASURE'] == 'T_CO2E')]
temp['REF_AREA'] = temp['REF_AREA'].replace('WXD', 'ROW')

# make lookup for aggregation
lookup_country = lookup['countries'][['oecd_code', 'combined_name']].dropna(how='any').drop_duplicates()
lookup_country = dict(zip(lookup_country['oecd_code'], lookup_country['combined_name']))

lookup_sectors = lookup['sectors'][['oecd_code', 'combined_name']].dropna(how='any').drop_duplicates()
lookup_sectors = dict(zip(lookup_sectors['oecd_code'], lookup_sectors['combined_name']))

lookup_fd = lookup['final_demand'][['oecd_code', 'combined_name']].dropna(how='any').drop_duplicates()
lookup_fd = dict(zip(lookup_fd['oecd_code'], lookup_fd['combined_name']))


for year in years:
    
    oecd_data[year] = {}

    name = mrio_filepath + 'ICIO/Ed_2024/' + str(year) + '_SML.csv'         
    icio = pd.read_csv(name, index_col=0)
    
    # save fs cats to filter out Y 
    Z_cols = oecd_lookup_cols.loc[oecd_lookup_cols['Industry/Final demand'].isin(oecd_lookup_industry['Code.1']) == True]['Sector code']
    Z_rows = oecd_lookup_rows.loc[oecd_lookup_rows['Industry/Final demand'].isin(oecd_lookup_industry['Code.1']) == True]['Sector code']
    
    Y_cols = cp.copy(oecd_lookup_cols)
    Y_cols['Ind'] = [str(x).split('_')[-1] for x in Y_cols['Sector code']]
    Y_cols = Y_cols.loc[Y_cols['Ind'].isin(oecd_lookup_fd) == True]['Sector code']

    oecd_data[year][footprint] = temp.loc[(temp['TIME_PERIOD'] == year) &
                                      (temp['REF_AREA'].isin(oecd_lookup_country['Code']) == True) &
                                      (temp['ACTIVITY'].isin(oecd_lookup_industry['Code.1']) == True)]
    oecd_data[year][footprint]['Sector code'] = oecd_data[year][footprint]['REF_AREA'] + '_' + oecd_data[year][footprint]['ACTIVITY']
    oecd_data[year][footprint] = oecd_data[year][footprint].set_index(['Sector code'])[['OBS_VALUE']]
    oecd_data[year][footprint] = oecd_data[year][footprint]*1000 # adjust unit to match Figaro
    oecd_data[year][footprint] = oecd_data[year][footprint].loc[Z_cols]

    oecd_data[year]['Z'] = icio.loc[Z_rows, Z_cols]
    oecd_data[year]['Y'] = icio.loc[Z_rows, Y_cols]
    
    # aggregate industries and products
    oecd_data[year]['Z'].index = pd.MultiIndex.from_arrays([[x.split('_')[0] for x in oecd_data[year]['Z'].index], ['_'.join(x.split('_')[1:]) for x in oecd_data[year]['Z'].index]])
    oecd_data[year]['Z'].columns = pd.MultiIndex.from_arrays([[x.split('_')[0] for x in oecd_data[year]['Z'].columns], ['_'.join(x.split('_')[1:]) for x in oecd_data[year]['Z'].columns]])
    oecd_data[year]['Z'] = oecd_data[year]['Z'].rename(columns=lookup_country, index=lookup_country)\
        .rename(columns=lookup_sectors, index=lookup_sectors).sum(axis=0, level=[0,1]).sum(axis=1, level=[0,1])
        
    oecd_data[year]['Y'].index = pd.MultiIndex.from_arrays([[x.split('_')[0] for x in oecd_data[year]['Y'].index], ['_'.join(x.split('_')[1:]) for x in oecd_data[year]['Y'].index]])
    oecd_data[year]['Y'].columns = pd.MultiIndex.from_arrays([[x.split('_')[0] for x in oecd_data[year]['Y'].columns], ['_'.join(x.split('_')[1:]) for x in oecd_data[year]['Y'].columns]])
    oecd_data[year]['Y'] = oecd_data[year]['Y'].rename(columns=lookup_country, index=lookup_country).rename(columns=lookup_fd, index=lookup_sectors).sum(axis=0, level=[0,1]).sum(axis=1, level=[0,1])
    
    oecd_data[year][footprint].index = pd.MultiIndex.from_arrays([[x.split('_')[0] for x in oecd_data[year][footprint].index], ['_'.join(x.split('_')[1:]) for x in oecd_data[year][footprint].index]])
    oecd_data[year][footprint] = oecd_data[year][footprint].rename(index=lookup_country).rename(index=lookup_sectors).sum(axis=0, level=[0,1])

# calculate oecd footprint
co2_oecd = {}
for year in years:
    Z = oecd_data[year]['Z']; Y = oecd_data[year]['Y']; stressor = oecd_data[year][footprint]
    co2_oecd[year] = cef.indirect_footprint_Z(Z, Y, stressor)
    
pickle.dump(co2_oecd, open(emissions_filepath + 'ICIO/ICIO_emissions_' + footprint + '_v' + version + '_agg_all.p', 'wb'))

print('ICIO Done')

############
## Gloria ##
############

co2_gloria = {}

# make lookup for aggregation
lookup_country = lookup['countries'][['gloria_code', 'combined_name']].dropna(how='any').drop_duplicates()
lookup_country = dict(zip(lookup_country['gloria_code'], lookup_country['combined_name']))

temp = lookup['sectors'][['gloria', 'combined_name']].dropna(how='any').drop_duplicates()
temp2 = cp.copy(temp)
temp2['gloria'] = ' ' + temp2['gloria'] + ' industry'
temp['gloria'] = ' ' + temp['gloria'] + ' product'
lookup_sectors = temp.append(temp2)
lookup_sectors = dict(zip(lookup_sectors['gloria'], lookup_sectors['combined_name']))

lookup_fd = lookup['final_demand'][['oecd_code', 'combined_name']].dropna(how='any').drop_duplicates()
lookup_fd = dict(zip(lookup_fd['oecd_code'], lookup_fd['combined_name']))

# read config file to get filenames
config_file= wd + 'ESCoE_Project/data/MRIO/Gloria/config_large.cfg'
gloria_filepath, outdir, lookup_filepath, labels_fname, lookup_fname, Z_fname, Y_fname, co2_fname, gloria_version = cef_g.read_config(config_file)

z_idx, industry_idx, product_idx, iix,pix,y_cols, sat_rows = cef_g.get_metadata_indices(gloria_filepath, lookup_filepath, labels_fname, lookup_fname)

# use this to extract correct row from stressor dataset below. Only one row from this DF is needed in the analysis
stressor_cat = "'GHG_total_EDGAR_consistent'" 

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
    # aggregate industries and products
    S = S.rename(columns=lookup_country, index=lookup_country).rename(columns=lookup_sectors, index=lookup_sectors).sum(axis=0, level=[0,1]).sum(axis=1, level=[0,1])
    U = U.rename(columns=lookup_country, index=lookup_country).rename(columns=lookup_sectors, index=lookup_sectors).sum(axis=0, level=[0,1]).sum(axis=1, level=[0,1])
    Y = Y.rename(columns=lookup_country, index=lookup_country).rename(columns=lookup_fd, index=lookup_sectors).sum(axis=0, level=[0,1]).sum(axis=1, level=[0,1])
    stressor = stressor.rename(columns=lookup_country).rename(columns=lookup_sectors).sum(axis=1, level=[0,1])

    co2_gloria[year] = cef_g.indirect_footprint_SUT_new(S, U, Y, stressor).T
    
    print('Gloria calculated for ' + str(year))

pickle.dump(co2_gloria, open(emissions_filepath + 'Gloria/Gloria_emissions_' + footprint + '_v' + version + '_agg_all.p', 'wb'))

print('Gloria Done')


##############
## Save all ##
##############

co2_all = {'exio':co2_exio,
           'figaro':co2_figaro,
           'gloria':co2_gloria,
           'oecd':co2_oecd}

pickle.dump(co2_all, open(emissions_filepath + 'Emissions_aggregated_all_agg_all.p', 'wb'))
