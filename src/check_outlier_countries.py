# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 10:30:46 2023

@author: geolki
"""

import pandas as pd
from sys import platform
import pickle
import numpy as np
import copy as cp
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
import matplotlib.pyplot as plt

# set working directory
# make different path depending on operating system
if platform[:3] == 'win':
    wd = 'O://'
else:
    wd = r'/Volumes/a72/' 
    
# define filepaths
data_filepath = wd + 'ESCoE_Project/data/'
emissions_filepath = wd + 'ESCoE_Project/data/Emissions/'
outputs_filepath = wd + 'ESCoE_Project/outputs/compare_all_outputs/'

# Dictonaries
country_list = ['Luxembourg', 'Malta', 'Ireland']
data_dict = {'oecd':'ICIO', 'exio':'Exio. 3.8.2', 'exio394':'Exio. 3.9.6', 'figaro':'Figaro','gloria':'Gloria'}

##################
## Run Analysis ##
##################

# load data
ghg_dict = pickle.load(open(emissions_filepath + 'Emissions_industry_ghg_all_agg_after.p', 'rb'))
ghg_dict['exio394'] = pickle.load(open(emissions_filepath + 'Emissions_industry_ghg_exio394_agg_after.p', 'rb'))['exio394']

ghg_full = pd.DataFrame()
for data in list(ghg_dict.keys()):
    for yr in list(ghg_dict[data].keys()):
        temp = ghg_dict[data][yr][country_list]
        temp[('Meta', 'dataset')] = data
        temp[('Meta', 'year')] = yr
        temp = temp.loc[temp[('Meta', 'year')] < 2021]
        temp = temp.set_index([('Meta', 'dataset'), ('Meta', 'year')], append=True)
        
        ghg_full = ghg_full.append(temp.fillna(0))
        
ghg_full = ghg_full.fillna(0).sum(axis=1, level=0)

# import country
by_import_country = ghg_full.sum(axis=0, level=[0, 2, 3]).unstack(level=1).stack(level=0)
by_import_country = by_import_country.apply(lambda x: (x/by_import_country.mean(1))*100)
by_import_country = by_import_country.stack().reset_index().rename(columns={'level_0':'Emission origin', ('Meta', 'year'):'year', ('Meta', 'dataset'):'dataset', 'level_2':'Consumer country', 0:'ghg difference'})

fig, axs = plt.subplots(ncols=3, figsize=(10, 18), sharey=True)
for i in range(3):
    item = by_import_country['Consumer country'].unique()[i]
    temp = by_import_country.loc[by_import_country['Consumer country'] == item]
    sns.pointplot(ax=axs[i], data=temp, y='Emission origin', x='ghg difference', hue='dataset', 
                  linestyles='', dodge=0.25, errorbar='sd')
    axs[i].axvline(100, c='k')
    axs[i].set_title(item)
    axs[i].set_ylabel('')

# industry
by_industry = ghg_full.sum(axis=0, level=[1, 2, 3]).unstack(level=1).stack(level=0)
by_industry = by_industry.apply(lambda x: (x/by_industry.mean(1))*100)
by_industry = by_industry.stack().reset_index().rename(columns={'level_0':'Emission origin', ('Meta', 'year'):'year', ('Meta', 'dataset'):'dataset', 'level_2':'Consumer country', 0:'ghg difference'})

fig, axs = plt.subplots(ncols=3, figsize=(10, 18), sharey=True)
for i in range(3):
    item = by_industry['Consumer country'].unique()[i]
    temp = by_industry.loc[by_industry['Consumer country'] == item]
    sns.pointplot(ax=axs[i], data=temp, y='Emission origin', x='ghg difference', hue='dataset', 
                  linestyles='', dodge=0.25, errorbar='sd')
    axs[i].axvline(100, c='k')
    axs[i].set_title(item)
    axs[i].set_ylabel('')

# year
by_year = ghg_full.sum(axis=0, level=[2, 3]).unstack(level=0).stack(level=0)
by_year = by_year.apply(lambda x: (x/by_year.mean(1))*100)
by_year = by_year.stack().reset_index().rename(columns={('Meta', 'year'):'Emission origin', ('Meta', 'dataset'):'dataset', 'level_1':'Consumer country', 0:'ghg difference'})

fig, axs = plt.subplots(ncols=3, figsize=(10, 18), sharey=True)
for i in range(3):
    item = by_year['Consumer country'].unique()[i]
    temp = by_year.loc[by_year['Consumer country'] == item]
    sns.scatterplot(ax=axs[i], data=temp, y='Emission origin', x='ghg difference', hue='dataset')
    axs[i].axvline(100, c='k')
    axs[i].set_title(item)
    axs[i].set_ylabel('')



##### analyse all
'''
by_all = ghg_full.unstack(level=2).stack(level=0).fillna(0)
by_all['mean'] = by_all.mean(1)

by_all_diff = ghg_full.unstack(level=2).stack(level=0).fillna(0)
by_all_diff = by_all_diff.apply(lambda x: x-by_all_diff.mean(1))
'''
temp = ghg_full.sum(axis=0, level=[2, 3]).stack().reset_index()
temp.columns = ['dataset', 'year', 'Consumer country', 'ghg_total']

by_all = ghg_full.stack().reset_index()
by_all.columns = ['Producer country', 'Industry', 'dataset', 'year', 'Consumer country', 'ghg']

by_all = by_all.merge(temp, on=['dataset', 'year', 'Consumer country'])
by_all = by_all.set_index(['dataset', 'Producer country', 'Industry', 'year', 'Consumer country'])
by_all['ghg_prop'] = by_all['ghg'] / by_all['ghg_total'] * 100

by_all = by_all.unstack(level='dataset').fillna(0)

## prop

cut_off = 3
by_ghg_prop = by_all['ghg_prop']
by_ghg_prop['prop_mean'] = by_ghg_prop.mean(1)

prop_ind = by_ghg_prop.groupby(['Industry', 'Consumer country', 'year']).sum().reset_index().groupby(['Industry', 'Consumer country']).mean()[['prop_mean']]
prop_count = by_ghg_prop.groupby(['Producer country', 'Consumer country', 'year']).sum().reset_index().groupby(['Producer country', 'Consumer country']).mean()[['prop_mean']]

by_ghg_prop = by_ghg_prop.join(prop_ind.rename(columns={'prop_mean':'from_ind'}))
by_ghg_prop = by_ghg_prop.join(prop_count.rename(columns={'prop_mean':'from_count'}))

temp_save = cp.copy(by_ghg_prop)[['from_ind', 'from_count']]

'''
by_ghg_prop['ind'] = [x[3] for x in by_ghg_prop.index.tolist()]
by_ghg_prop['og_c'] = [x[2] for x in by_ghg_prop.index.tolist()]

by_ghg_prop.loc[by_ghg_prop['from_ind'] < cut_off, 'ind'] = 'Other industries'
by_ghg_prop.loc[by_ghg_prop['from_count'] < cut_off, 'og_c'] = 'Other country'

by_ghg_prop = by_ghg_prop.groupby(['og_c', 'ind', 'year', 'Consumer country']).sum().drop(['from_ind', 'from_count'], axis=1)

by_ghg_prop = by_ghg_prop.drop(['prop_mean'], axis=1).unstack('Consumer country').stack('dataset')

for country in country_list:
    for item in ['ind', 'og_c']:
        temp = by_ghg_prop[[country]].dropna(how='all').unstack('dataset')
        temp[(country, 'zz')] = 0
        temp = temp.stack('dataset')
        temp = temp.groupby([item, 'year', 'dataset']).sum().reset_index()
        temp['yr_data'] = temp['year'].astype(str) + '_' + temp['dataset'] 
        
        temp = temp.drop(['year', 'dataset'], axis=1).set_index([item, 'yr_data']).unstack(item).fillna(0).droplevel(axis=1, level=0)
        
        temp.plot(kind='bar', stacked=True, figsize=(20, 10))
        plt.title(country)
        plt.show()
''' 
## value

cut_off =3
by_ghg_value = by_all['ghg']
by_ghg_value['ghg_mean'] = by_ghg_value.mean(1)

by_ghg_value = by_ghg_value.join(temp_save)

by_ghg_value['ind'] = [x[1] for x in by_ghg_value.index.tolist()]
by_ghg_value['og_c'] = [x[0] for x in by_ghg_value.index.tolist()]

by_ghg_value.loc[by_ghg_value['from_ind'] < cut_off, 'ind'] = 'Other industries'
by_ghg_value.loc[by_ghg_value['from_count'] < cut_off, 'og_c'] = 'Other country'

by_ghg_value = by_ghg_value.groupby(['og_c', 'ind', 'year', 'Consumer country']).sum().drop(['from_ind', 'from_count'], axis=1)

by_ghg_value = by_ghg_value.drop(['ghg_mean'], axis=1).unstack('Consumer country').stack('dataset')

'''
for country in country_list:
    for item in ['ind', 'og_c']:
        temp = by_ghg_value[[country]].dropna(how='all').unstack('dataset')
        
        order = temp.groupby(item).sum().mean(1).sort_values(ascending=False).index.tolist()
        
        temp[(country, 'zz')] = 0
        temp = temp.stack('dataset')
        temp = temp.groupby([item, 'year', 'dataset']).sum().reset_index()
        temp['yr_data'] = temp['year'].astype(str) + '_' + temp['dataset'] 
        
        temp = temp.drop(['year', 'dataset'], axis=1).set_index([item, 'yr_data']).unstack(item).fillna(0).droplevel(axis=1, level=0)
        
        temp[order].plot(kind='bar', stacked=True, figsize=(20, 10))
        plt.title(country)
        plt.show()
'''      
        
for country in country_list:
    for item in ['ind', 'og_c']:
        temp = by_ghg_value[[country]].dropna(how='all')
        temp.loc[temp[country] < 0, country] = 0
        
        temp = temp.groupby([item, 'year', 'dataset']).sum().unstack('dataset').droplevel(axis=1, level=0)
        
        order = order = temp.groupby(item).mean().mean(1).sort_values(ascending=False).index.tolist()
        
        for ds in temp.columns.tolist():
            plot_data = temp[[ds]].unstack('year').loc[order].T.droplevel(axis=0, level=0)
       
            plot_data.plot(kind='area', stacked=True, figsize=(5, 5), cmap='tab20')
            plt.title(country + ' ' + ds)
            plt.legend(bbox_to_anchor=(1, 1))
            plt.show()
         
     
