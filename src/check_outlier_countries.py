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
from matplotlib.ticker import MaxNLocator

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
plot_filepath = 'C:/Users/geolki/OneDrive - University of Leeds/Leeds onedrive/Postdoc/ESCoE/plots/'


# Dictonaries
country_list = ['Luxembourg', 'Malta', 'Ireland']
data_dict = {'oecd':'ICIO', 'exio':'Exio. 3.8.2', 'exio394':'Exio. 3.9.6', 'figaro':'Figaro','gloria':'Gloria'}


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

##################
## Run Analysis ##
##################

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

## value

cut_off = 3
exclude_c = ['Denmark']

by_ghg_value = by_all['ghg']
by_ghg_value['ghg_mean'] = by_ghg_value.mean(1)

by_ghg_value = by_ghg_value.join(temp_save)

by_ghg_value['ind'] = [x[1] for x in by_ghg_value.index.tolist()]
by_ghg_value['og_c'] = [x[0] for x in by_ghg_value.index.tolist()]

by_ghg_value.loc[by_ghg_value['from_ind'] < cut_off, 'ind'] = 'Other industries'
by_ghg_value.loc[(by_ghg_value['from_count'] < cut_off) & (by_ghg_value['og_c'].isin(exclude_c) == False), 'og_c'] = 'Other countries'

by_ghg_value = by_ghg_value.groupby(['og_c', 'ind', 'year', 'Consumer country']).sum().drop(['from_ind', 'from_count'], axis=1)

by_ghg_value = by_ghg_value.drop(['ghg_mean'], axis=1).unstack('Consumer country').stack('dataset')



check_L = by_ghg_value[['Malta']].dropna(how='any').unstack('year').droplevel(axis=1, level=0)
check_L2 = cp.copy(check_L).drop(2010, axis=1)
years = range(2010, 2021)
for yr in years[1:]:
    check_L2[yr] = check_L[yr] - check_L[yr-1]
check_L3 = pd.DataFrame(check_L2.mean(1)).unstack('dataset')

check_L4 = check_L.groupby(['ind', 'dataset']).sum().mean(1).unstack(level='dataset')

for item in ['ind', 'og_c']:
    for c in range(3):
        country = country_list[c]
        
        temp = by_ghg_value[[country]].dropna(how='all')
        temp.loc[temp[country] < 0, country] = 0
        
        temp = temp.groupby([item, 'year', 'dataset']).sum().unstack('dataset').droplevel(axis=1, level=0)
        
        order = order = temp.groupby(item).mean().mean(1).sort_values(ascending=False).index.tolist()

        fig, axs = plt.subplots(ncols=5, sharey=True, figsize=(20, 3))

        for i in range(5):
            ds = temp.columns.tolist()[i]
            plot_data = temp[[ds]].reset_index() #.rename(column:{'og_c':'Producer country'}) # .unstack('year').loc[order].T.droplevel(axis=0, level=0)
        
            sns.lineplot(ax=axs[i], data=plot_data, x='year', y=ds, hue=item, palette='tab20')
            axs[i].set_title(data_dict[ds])
            axs[i].set_ylabel(country + ' Footprint (ktCO\N{SUBSCRIPT TWO}e)')
            axs[i].set_xlabel('Year')
            axs[i].xaxis.set_major_locator(MaxNLocator(integer=True))
            
            if i == 4:
                if item == 'ind':
                    axs[i].legend(bbox_to_anchor=(-0.5, -0.175), ncol=3)
                else:
                    axs[i].legend(bbox_to_anchor=(-0.8, -0.175), ncol=10)
            else:
                axs[i].legend().remove()
        
        plt.savefig(plot_filepath + 'Lineplot_detail_' + country + '_' + item + '.png', dpi=200, bbox_inches='tight')
        plt.show()
     
 
