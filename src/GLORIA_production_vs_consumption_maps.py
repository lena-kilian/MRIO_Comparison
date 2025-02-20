# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:31:51 2023

@author: geolki
"""

import pandas as pd
from sys import platform
import calculate_emissions_functions_gloria as cef_g
import matplotlib.pyplot as plt
import geopandas as gpd
import seaborn as sns
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

version = '2024'
footprint = 'ghg'
years = range(2018, 2024)
    

############
## Gloria ##
############

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')).set_index('name')[['pop_est']]
wrld_lookup = pd.read_excel('C:/Users/geolki/OneDrive - University of Leeds/Applications/Fellowships/Wellcome Trust/presentation_map_lookup_v2.xlsx')
wrld_lookup = wrld_lookup.set_index('geopandas').join(world, how='outer')
wrld_lookup = wrld_lookup[['Gloria', 'Income level', 'pop_est']].dropna(how='any')

country_lookup = pd.read_excel('O:/ESCoE_Project/data/MRIO/Gloria/GLORIA_ReadMe_059a.xlsx', sheet_name='Regions', index_col=1)[['Region_names']]

# define sample year, normally this is: range(2010, 2019)
# here years is now determined from inputs,
# it used to be a range(2010, 2019). In future work this will likley be range(2001, 2023)

results_origin = pd.DataFrame()

results_pct = pd.DataFrame()

results_daly = pd.DataFrame()

for year in years:
    #ghg_all = pd.read_csv('C:/Users/geolki/OneDrive - University of Leeds/Postdoc/Gloria_detail/GHG/Gloria_ghg_industries_' + str(year) + '_all.csv', header=[0, 1], index_col=[0, 1, 2])
    #ghg_all.columns = [x[0].split('.')[0] for x in ghg_all.columns]
    

    consumption1 = pd.read_csv('C:/Users/geolki/OneDrive - University of Leeds/Postdoc/DALY/data/outputs/co2_excl_short_cycle_org_c_' + str(year) + '.csv', header=[0, 1], index_col=[0, 1])
    
    consumption=cp.copy(consumption1)
    consumption.index.names = ['final_demand_country', 'final_demand_cat']
    consumption.columns.names = ['daly_origin_country', 'daly_origin_industry']
    
    consumption = consumption.sum(axis=0, level=0).sum(axis=1, level=0).stack().reset_index()
    consumption['final_demand_country'] = consumption['final_demand_country'].map(dict(zip(country_lookup.reset_index()['Region_acronyms'], country_lookup['Region_names'])))
    consumption['final_demand_country'] = consumption['final_demand_country'].map(dict(zip(wrld_lookup['Gloria'], wrld_lookup['Income level'])))
    
    consumption['daly_origin_country'] = consumption['daly_origin_country'].map(dict(zip(country_lookup.reset_index()['Region_acronyms'], country_lookup['Region_names'])))
    consumption['daly_origin_country'] = consumption['daly_origin_country'].map(dict(zip(wrld_lookup['Gloria'], wrld_lookup['Income level'])))
    
    consumption_og = consumption.groupby(['final_demand_country', 'daly_origin_country']).sum().unstack('daly_origin_country')
    consumption_og = consumption_og.apply(lambda x: x/x.sum()*100)
    
    consumption_og = consumption_og.stack().reset_index()
    consumption_og['emission_destination'] = 'Domestic'
    consumption_og.loc[consumption_og['final_demand_country'] != consumption_og['daly_origin_country'], 'emission_destination'] = 'Exported'
    consumption_og = consumption_og.groupby(['emission_destination', 'daly_origin_country']).sum()
    
    consumption_og = consumption_og.unstack('daly_origin_country').droplevel(axis=1, level=0)[['High', 'Upper-middle', 'Lower-middle', 'Lower']]
    consumption_og['year'] = year
    results_origin = results_origin.append(consumption_og.reset_index())
    
    consumption=cp.copy(consumption1)
    consumption = pd.DataFrame(consumption.sum(axis=0, level=0).sum(axis=1)); consumption.columns = ['consumption']
    
    production = pd.read_csv('C:/Users/geolki/OneDrive - University of Leeds/Postdoc/DALY/data/outputs/co2_excl_short_cycle_org_c_' + str(year) + '_production.csv', header=[0, 1], index_col=0)
    production = production.T.sum(axis=0, level=0); production.columns = ['production']
    
    daly = consumption.join(production).join(country_lookup).set_index('Region_names')
    
    daly_small = daly.join(wrld_lookup.set_index('Gloria'), how='inner').dropna(how='any').sum(axis=0, level=0)
    #daly_small = daly_small.groupby('Income level').sum()
    
    daly_total = daly_small.groupby('Income level').sum().loc[['High', 'Upper-middle', 'Lower-middle', 'Lower']]
    daly_total['year'] = year
    results_daly = results_daly.append(daly_total.reset_index())
    
    #daly_small[['production', 'consumption']] = daly_small[['production', 'consumption']].apply(lambda x: x/daly_small['pop_est']*1000)
    
    plot_data = daly_small.set_index('Income level', append=True)[['consumption', 'production']].stack().reset_index().rename(columns={0:'Value'})
    
    sns.barplot(data=plot_data, x='Income level', y='Value', hue='level_2')
    
    daly_pct = daly_small.groupby('Income level').sum()
    daly_pct = daly_pct.apply(lambda x: x/x.sum() * 100)
    
    daly_pct['year'] = year
    
    daly_pct = daly_pct.loc[['High', 'Upper-middle', 'Lower-middle', 'Lower']]
    
    results_pct = results_pct.append(daly_pct)
    
    
    daly_pct.T[['High', 'Upper-middle', 'Lower-middle', 'Lower']].drop('year').plot(kind='bar', stacked=True) # [['consumption', 'pop_est']]
    
    #plt.savefig('C:/Users/geolki/OneDrive - University of Leeds/Applications/Fellowships/Wellcome Trust/consumptopnvsproduction_map_' + str(year) + '.png', dpi=200, bboxinches='tight')
    

    print(year)
    

plot_data = results_pct.set_index('year', append=True).stack().reset_index()
sns.lineplot(data=plot_data, x='year', y=0, hue='Income level' ,style='level_2')


plot_data = results_origin.set_index('emission_destination').loc['Exported'].set_index('year').stack().reset_index()
sns.lineplot(data=plot_data, x='year', y=0, hue='daly_origin_country')


summary_pct = results_pct.mean(axis=0, level=0)
summary_origin = results_origin.set_index(['emission_destination', 'year']).loc['Exported'].mean()
summary_daly = results_daly.set_index(['Income level', 'year'])
summary_daly = summary_daly.apply(lambda x: x/summary_daly['pop_est'] *1000)

plot_data = summary_daly.drop('pop_est', axis=1).stack().reset_index()
sns.lineplot(data=plot_data, x='year', y=0, hue='Income level' ,style='level_2')

sns.barplot(data=plot_data, x='Income level', y=0, hue='level_2')

summary_daly = summary_daly.mean(axis=0, level=0)

