# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 07:41:09 2025

@author: geolki
"""

import pandas as pd

years = range(2010, 2022)

# import files
# read conversion file
co2e_conversion = pd.read_csv('O:/EXIOBASE/EXIOBASE310f_CCf_list.csv')
# read extensions
extensions = {}
for year in years:
    extensions[year] = pd.read_csv('O:/EXIOBASE/3.9.6 ixi/' + str(year) + '/air_emissions/F.txt')


# filter for AR5
co2e_conversion = co2e_conversion.loc[co2e_conversion['impact'] == 'GHG emissions AR5 (GWP100)|kg CO2 eq.||GWP100 (IPCC, 2010)']



