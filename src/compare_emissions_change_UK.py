# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 10:30:46 2023

@author: geolki
"""

import pandas as pd
from sys import platform
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# set working directory
# make different path depending on operating system
if platform[:3] == 'win':
    wd = 'O://'
else:
    wd = r'/Volumes/a72/' 

# define filepaths
data_filepath = wd + 'ESCoE_Project/data/'
emissions_filepath = wd + 'ESCoE_Project/data/Emissions/'
plot_filepath = wd + 'ESCoE_Project/outputs/compare_all_outputs/plots/'


co2_all = pickle.load(open(emissions_filepath + 'Emissions_aggregated_all.p', 'rb'))
    

datasets = list(co2_all.keys())
years = list(co2_all[datasets[0]].keys())

data_comb = ['oecd, figaro', 'oecd, exio', 'oecd, gloria', 
             'figaro, exio', 'figaro, gloria',
             'exio, gloria']


###############
## Summarise ##
###############

for country in co2_all[datasets[0]][years[0]].columns.levels[0].tolist():
    total = pd.DataFrame()
    for year in years:
        temp_oecd = pd.DataFrame(co2_all['oecd'][year][[country]].sum(axis=1, level=0)).rename(columns={country:'oecd'})
        temp_figaro = pd.DataFrame(co2_all['figaro'][year][[country]].sum(axis=1, level=0)).rename(columns={country:'figaro'})
        temp_gloria = pd.DataFrame(co2_all['gloria'][year][[country]].sum(axis=1, level=0)).rename(columns={country:'gloria'})
        temp_exio = pd.DataFrame(co2_all['exio'][year][[country]].sum(axis=1, level=0)).rename(columns={country:'exio'})
        
        temp = temp_oecd.join(temp_figaro).join(temp_gloria).join(temp_exio)
        
        temp['year'] = year
        
        total = total.append(temp.reset_index())
        
    total = total.rename(columns={'level_0':'country', 'level_1':'sector'}).set_index(['country', 'sector', 'year']).fillna(0)
    
    domestic = total.loc[country, :]
    imports = total.drop(country)
    
    for item in ['domestic']: # ['total', 'domestic', 'imports']:
        data = eval(item).reset_index().groupby(['year']).sum()[datasets]
        data.plot()
        plt.title(country + ' ' + item)
        plt.savefig(plot_filepath + 'UK_lineplot_country_emissions_' + item + '.png', dpi=200, bbox_inches='tight')

#####################
## Change in trend ##
#####################

temp = total.unstack(level=1).stack(level=0)
change = temp[years[1:]]
for year in years[1:]:
    change[year] = temp[year] / temp[year - 1]
    
change = change.reset_index().rename(columns={'level_1':'dataset'})

# Compare all

change2 = change#.loc[change['dataset'] != 'gloria']

change3 = change2.groupby('country').describe().stack(level=0)[['min', 'max', 'mean']]
change3['range'] = change3['max'] - change3['min']
change3['Same_direction'] = False
change3.loc[((change3['min']>1) & (change3['max']>1) |
             (change3['min']<1) & (change3['max']<1) |
             (change3['min']==1) & (change3['max']==1)), 'Same_direction'] = True

temp = change3[['Same_direction', 'range']].reset_index()
sns.boxplot(data=temp, x='country', y='range', hue='Same_direction', showfliers=False); plt.xticks(rotation=90); plt.show()

change4 = change3.reset_index().groupby(['country', 'Same_direction']).describe()['range'][['count', 'min', 'max']].unstack(level=1).fillna(0)

# Compare pariwise

change5 = change.set_index(['country', 'dataset']).unstack('dataset').stack('year')
change5.columns = pd.MultiIndex.from_arrays([['dataset']*len(datasets), change5.columns.tolist()])

for pair in data_comb:
    ds0 = pair.split(', ')[0]; ds1 = pair.split(', ')[1]
    change5[('Same_direction', ds0 + '_' + ds1)] = False
    change5.loc[((change5[('dataset', ds0)]>1) & (change5[('dataset', ds1)]>1) |
                 (change5[('dataset', ds0)]<1) & (change5[('dataset', ds1)]<1) |
                 (change5[('dataset', ds0)]==1) & (change5[('dataset', ds1)]==1)), 
                ('Same_direction', ds0 + '_' + ds1)] = True
    change5[('Abs_diff', ds0 + '_' + ds1)] = np.abs(change5[('dataset', ds0)] - change5[('dataset', ds1)])
    
count = change5[['Same_direction']].stack(level=1).reset_index().rename(columns={'level_2':'dataset'}).groupby(['country', 'dataset', 'Same_direction']).count().unstack('Same_direction').droplevel(axis=1, level=0).fillna(0)
count['pct_same'] = count[True] / (count[True] + count[False])*100
change6 = change5[['Abs_diff']].unstack('country').describe().T[['min', 'max', 'mean', 'std']].droplevel(axis=0, level=0)
change6.index.names = ['dataset', 'country']

change6 = change6.join(count[['pct_same']])

temp = change5['Abs_diff'].stack().reset_index().rename(columns={'level_2':'dataset', 0:'Abs_diff'})
order = temp.groupby('country').mean().sort_values('Abs_diff', ascending=True)
temp = temp.set_index(['country', 'year']).loc[order.index.tolist()].reset_index()
fig, ax = plt.subplots(figsize=(25,5))
sns.boxplot(ax=ax, data=temp, x='country', y='Abs_diff', hue='dataset', showfliers=False); 
plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1)); 
plt.savefig(plot_filepath + 'UK_boxplot_country_difference.png', dpi=200, bbox_inches='tight')

total_diff = order[['Abs_diff']]
total_diff['Under'] = 'Other'
for i in [50, 20, 10, 5, 1]:
    total_diff.loc[total_diff['Abs_diff'] <= i/100, 'Under'] = str(i) + '%'

sns.barplot(data=change6.reset_index(), x='country', y='pct_same', hue='dataset'); plt.xticks(rotation=90); plt.show()

change_country = change6.mean(axis=0, level='country').sort_values('pct_same', ascending=False)
change_data = change6.mean(axis=0, level='dataset').sort_values('pct_same', ascending=False)

sns.boxplot(data=change6.swaplevel(axis=0).loc[change_country.index.tolist()].reset_index(), x='country', y='pct_same'); plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1));
plt.savefig(plot_filepath + 'UK_boxplot_country_pctsame_bycountry.png', dpi=200, bbox_inches='tight')

sns.boxplot(data=change6.loc[change_data.index.tolist()].reset_index(), x='dataset', y='pct_same'); plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1)); plt.show()
plt.savefig(plot_filepath + 'UK_boxplot_country_pctsame_bydatapair.png', dpi=200, bbox_inches='tight')

sns.scatterplot(data=change6.swaplevel(axis=0).loc[change_country.index.tolist()].reset_index(), x='country', y='pct_same', hue='dataset'); plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1)); plt.show()
plt.savefig(plot_filepath + 'UK_scatterlot_country_pctsame.png', dpi=200, bbox_inches='tight')
