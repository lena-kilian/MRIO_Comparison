# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 10:30:46 2023

@author: geolki
"""

import pandas as pd
from sys import platform
import pickle

# set working directory
# make different path depending on operating system
if platform[:3] == 'win':
    wd = 'O://'
else:
    wd = r'/Volumes/a72/' 

# define version
version = '2024'
footprint = 'ghg'

# define filepaths
data_filepath = wd + 'ESCoE_Project/data/'
emissions_filepath = wd + 'ESCoE_Project/data/Emissions/'

years = range(2010, 2021)

# Load data
co2_gloria = {year: pd.read_csv('O:/ESCoE_Project/data/Emissions/Gloria/Gloria_CO2_' + str(year) + '.csv', index_col=[0,1], header=[0, 1]) for year in years}# should be 2010
co2_oecd = pickle.load(open(emissions_filepath + 'ICIO/ICIO_emissions_' + footprint + '_v' + version + '.p', 'rb'))
co2_figaro = pickle.load(open(emissions_filepath + 'Figaro/Figaro_emissions_' + footprint + '_v' + version + '.p', 'rb'))
co2_exio = pickle.load(open(emissions_filepath + 'Exiobase/Exiobase_emissions.p', 'rb'))

# Load lookup file
lookup = pd.read_excel(data_filepath + 'lookups/mrio_lookup_sectors_countries_finaldemand.xlsx', sheet_name = None)

# convert to same categories
for year in years:
    print(year)
    
    ############
    ## Figaro ##
    ############
    
    # make dictionaries
    figaro_countries = lookup['countries'][['figaro_code', 'combined_name']].drop_duplicates();
    figaro_countries = dict(zip(figaro_countries['figaro_code'], figaro_countries['combined_name']))
    
    figaro_sectors = lookup['sectors'][['figaro_code', 'combined_name']].drop_duplicates();
    figaro_sectors = dict(zip(figaro_sectors['figaro_code'], figaro_sectors['combined_name']))
    
    figaro_fd = lookup['final_demand'][['figaro_code', 'combined_name']].drop_duplicates();
    figaro_fd = dict(zip(figaro_fd['figaro_code'], figaro_fd['combined_name']))
    
    # make new indices using combined names
    idx1 = []; idx2 = []
    for x in co2_figaro[year].index:
        c = x.split('_')[0]
        s = x[len(c)+1:]
        idx1.append(figaro_countries[c]); idx2.append(figaro_sectors[s]);
        
    col1 = []; col2 = []
    for x in co2_figaro[year].columns:
        c = x.split('_')[0]
        f = x[len(c)+1:]
        col1.append(figaro_countries[c]); col2.append(figaro_fd[f]);
    
    # rename indices
    co2_figaro[year].index = pd.MultiIndex.from_arrays([idx1, idx2])
    co2_figaro[year].columns = pd.MultiIndex.from_arrays([col1, col2])
    # aggregate
    co2_figaro[year] = co2_figaro[year].sum(axis=0, level=[0, 1]).sum(axis=1, level=[0, 1])
    
    print('Figaro', year, co2_figaro[year].sum().sum())
    
    
    ##########
    ## ICIO ##
    ##########
    
    # make dictionaries
    oecd_countries = lookup['countries'][['oecd_code', 'combined_name']].drop_duplicates();
    oecd_countries = dict(zip(oecd_countries['oecd_code'], oecd_countries['combined_name']))
    
    oecd_sectors = lookup['sectors'][['oecd_code', 'combined_name']].drop_duplicates();
    oecd_sectors = dict(zip(oecd_sectors['oecd_code'], oecd_sectors['combined_name']))
    
    oecd_fd = lookup['final_demand'][['oecd_code', 'combined_name']].drop_duplicates();
    oecd_fd = dict(zip(oecd_fd['oecd_code'], oecd_fd['combined_name']))
    
    # rename indices
    co2_oecd[year] = co2_oecd[year].rename(columns=oecd_countries).swaplevel(axis=1).rename(columns=oecd_fd).swaplevel(axis=1)\
        .rename(index=oecd_countries).swaplevel(axis=0).rename(index=oecd_sectors).swaplevel(axis=0)
   
    # aggregate
    co2_oecd[year] = co2_oecd[year].sum(axis=0, level=[0, 1]).sum(axis=1, level=[0, 1])
    
    print('ICIO', year, co2_oecd[year].sum().sum())

    
    ############
    ## Gloria ##
    ############
    
    # make dictionaries
    gloria_countries = lookup['countries'][['gloria_code', 'combined_name']].drop_duplicates();
    gloria_countries = dict(zip(gloria_countries['gloria_code'], gloria_countries['combined_name']))
    
    gloria_sectors = lookup['sectors'][['gloria', 'combined_name']].dropna(how='any').drop_duplicates();
    gloria_sectors = dict(zip(gloria_sectors['gloria'], gloria_sectors['combined_name']))
    
    gloria_fd = lookup['final_demand'][['gloria', 'combined_name']].drop_duplicates();
    gloria_fd = dict(zip(gloria_fd['gloria'], gloria_fd['combined_name']))
    
    # rename indices
    co2_gloria[year] = co2_gloria[year].T
    # sectors
    c = [x[0] for x in co2_gloria[year].index.tolist()]; s = [x[1][1:].replace(' industry', '') for x in co2_gloria[year].index.tolist()]
    co2_gloria[year].index = pd.MultiIndex.from_arrays([[gloria_countries[x] for x in c], [gloria_sectors[x] for x in s]])
    # fd
    c = [x[0] for x in co2_gloria[year].columns.tolist()]; f = [x[1][1:] for x in co2_gloria[year].columns.tolist()]
    co2_gloria[year].columns = pd.MultiIndex.from_arrays([[gloria_countries[x] for x in c], [gloria_fd[x] for x in f]])
    # aggregate
    co2_gloria[year] = co2_gloria[year].sum(axis=0, level=[0, 1]).sum(axis=1, level=[0, 1])
    
    
    print('Gloria', year, co2_gloria[year].sum().sum())

    ##########
    ## EXIO ##
    ##########
    
    # make dictionaries
    exio_countries = lookup['countries'][['exio_code', 'combined_name']].drop_duplicates();
    exio_countries = dict(zip(exio_countries['exio_code'], exio_countries['combined_name']))
    
    exio_sectors = lookup['sectors'][['exio', 'combined_name']].drop_duplicates();
    exio_sectors = dict(zip(exio_sectors['exio'], exio_sectors['combined_name']))
    
    exio_fd = lookup['final_demand'][['exio', 'combined_name']].drop_duplicates();
    exio_fd = dict(zip(exio_fd['exio'], exio_fd['combined_name']))
    
    # rename indices
    co2_exio[year] = co2_exio[year].T
    co2_exio[year].index = pd.MultiIndex.from_arrays([[exio_countries[x[0]] for x in co2_exio[year].index.tolist()], [exio_sectors[x[1]] for x in co2_exio[year].index.tolist()]])
    co2_exio[year].columns = pd.MultiIndex.from_arrays([[exio_countries[x[0]] for x in co2_exio[year].columns.tolist()], [exio_fd[x[1]] for x in co2_exio[year].columns.tolist()]])
    # aggregate
    co2_exio[year] = co2_exio[year].sum(axis=0, level=[0, 1]).sum(axis=1, level=[0, 1])
    co2_exio[year] = co2_exio[year] /1000000 # adjust to same unit as the others
    
    print('Exiobase', year, co2_exio[year].sum().sum())


# Save all
co2_all = {}
co2_all['gloria'] = co2_gloria
co2_all['exio'] = co2_exio
co2_all['figaro'] = co2_figaro
co2_all['oecd'] = co2_oecd

pickle.dump(co2_all, open(emissions_filepath + 'Emissions_aggregated_all.p', 'wb'))
