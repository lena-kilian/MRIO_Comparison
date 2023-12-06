# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 10:30:46 2023

@author: geolki
"""

import pandas as pd
from sys import platform
import calculate_emissions_functions as cef
import itertools
import pickle
import numpy as np
import os
import copy as cp


# set working directory
# make different path depending on operating system
if platform[:3] == 'win':
    wd = 'O://'
else:
    wd = r'/Volumes/a72/' 

# define filepaths
data_filepath = wd + 'ESCoE_Project/data/'
emissions_filepath = wd + 'ESCoE_Project/data/Emissions/'

years = range(2010, 2019)

# Load data
co2_gloria = {year: pd.read_csv('O:/ESCoE_Project/data/Emissions/Gloria/CO2_' + str(year) + '.csv') for year in years}
co2_oecd = pickle.load(open(emissions_filepath + 'ICIO/ICIO_emissions.p', 'rb'))
co2_figaro = pickle.load(open(emissions_filepath + 'Figaro/Figaro_emissions.p', 'rb'))
#co2_exio = pickle.load(open(emissions_filepath + 'ICIO/ICIO_emissions.p', 'rb'))

# Load lookup file
lookup = pd.read_excel(data_filepath + 'lookups/mrio_lookup_sectors_countries_finaldemand.xlsx', sheet_name = None)

# convert to same categories
for year in years:
    
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
    co2_oecd[year].index = pd.MultiIndex.from_arrays([[oecd_countries[x[0]] for x in co2_oecd[year].index.tolist()], [oecd_sectors[x[1]] for x in co2_oecd[year].index.tolist()]])
    co2_oecd[year].columns = pd.MultiIndex.from_arrays([[oecd_countries[x[0]] for x in co2_oecd[year].columns.tolist()], [oecd_fd[x[1]] for x in co2_oecd[year].columns.tolist()]])
    # aggregate
    co2_oecd[year] = co2_oecd[year].sum(axis=0, level=[0, 1]).sum(axis=1, level=[0, 1])

    
    ############
    ## Gloria ##
    ############
    
    # make dictionaries
    oecd_countries = lookup['countries'][['oecd_code', 'combined_name']].drop_duplicates();
    oecd_countries = dict(zip(oecd_countries['oecd_code'], oecd_countries['combined_name']))
    
    oecd_sectors = lookup['sectors'][['oecd_code', 'combined_name']].drop_duplicates();
    oecd_sectors = dict(zip(oecd_sectors['oecd_code'], oecd_sectors['combined_name']))
    
    oecd_fd = lookup['final_demand'][['oecd_code', 'combined_name']].drop_duplicates();
    oecd_fd = dict(zip(oecd_fd['oecd_code'], oecd_fd['combined_name']))
    
    # rename indices
    co2_oecd[year].index = pd.MultiIndex.from_arrays([[oecd_countries[x[0]] for x in co2_oecd[year].index.tolist()], [oecd_sectors[x[1]] for x in co2_oecd[year].index.tolist()]])
    co2_oecd[year].columns = pd.MultiIndex.from_arrays([[oecd_countries[x[0]] for x in co2_oecd[year].columns.tolist()], [oecd_fd[x[1]] for x in co2_oecd[year].columns.tolist()]])
    # aggregate
    co2_oecd[year] = co2_oecd[year].sum(axis=0, level=[0, 1]).sum(axis=1, level=[0, 1])


