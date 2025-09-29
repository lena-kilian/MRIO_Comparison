# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 07:41:09 2025

@author: geolki
"""

import pandas as pd

years = range(2010, 2022)
var_extension= 'GHG emissions AR5 (GWP100)|kg CO2 eq.||GWP100 (IPCC, 2010)'

# import files
# read conversion file
co2e_conversion = pd.read_csv('O:/EXIOBASE/EXIOBASE310f_CCf_list.csv', index_col=[1, 0])
# read extensions
extensions_raw = {}
for year in years:
    extensions_raw[year] = pd.read_csv('O:/EXIOBASE/3.9.6 ixi/' + str(year) + '/air_emissions/F.txt', sep='\t', header = [0, 1], index_col=0)


# filter for AR5
co2e_conversion = co2e_conversion.loc[var_extension]

# convert ghg extensions to co2e
extensions_ghg = {}
for year in years:
    temp = extensions_raw[year].loc[co2e_conversion.index.tolist()]
    
    for item in temp.index:
        temp.loc[item] = temp.loc[item] * co2e_conversion.loc[item, 'value']
        
    extensions_ghg[year] = pd.DataFrame(temp.sum())

'''
# test
for year in years:
    temp = pd.read_csv('O:/EXIOBASE/3.8.2/MRSUT_' + str(year) + '/F.txt',  sep='\t', header = [0, 1], index_col=0).loc[['GHG emissions AR5 (GWP100) | GWP100 (IPCC, 2010)']].T
    temp.columns = ['GHG_3.8.2']
    
    temp = temp.join(extensions_ghg[year].rename(columns={0:'GHG_3.9.6'}))
    
    print(year, temp.corr().loc['GHG_3.8.2', GHG_3.9.6])
'''

# reformat and save

for year in years:
    temp = extensions_ghg[year].rename(columns={0:var_extension}).T
    temp.to_csv('O:/EXIOBASE/3.9.6_processed_GHG/' + str(year) + '.csv')