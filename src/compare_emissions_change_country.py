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

summary = pd.DataFrame()
for year in years:
    temp_oecd = pd.DataFrame(co2_all['oecd'][year].sum(axis=1, level=0).sum(axis=0)).rename(columns={0:'oecd'})
    temp_figaro = pd.DataFrame(co2_all['figaro'][year].sum(axis=1, level=0).sum(axis=0)).rename(columns={0:'figaro'})
    temp_gloria = pd.DataFrame(co2_all['gloria'][year].sum(axis=1, level=0).sum(axis=0)).rename(columns={0:'gloria'})
    temp_exio = pd.DataFrame(co2_all['exio'][year].sum(axis=1, level=0).sum(axis=0)).rename(columns={0:'exio'})
    
    temp = temp_oecd.join(temp_figaro).join(temp_gloria).join(temp_exio)
    
    temp['year'] = year
    
    summary = summary.append(temp.reset_index())
    
summary = summary.rename(columns={'index':'country'}).set_index(['country', 'year'])

for country in summary.index.levels[0].tolist():
    summary.loc[country,:].plot()
    plt.title(country)
    plt.savefig(plot_filepath + 'lineplot_country_emissions_' + country + '.png', dpi=200, bbox_inches='tight')
    plt.show()
    
#####################
## Change in trend ##
#####################

temp = summary.unstack(level=1).stack(level=0)
change = temp[years[1:]]
for year in years[1:]:
    change[year] = temp[year] / temp[year - 1]
    
change = change.reset_index().rename(columns={'level_1':'dataset'})

# Compare all

change2 = change#.loc[change['dataset'] != 'gloria']

change_plot = (change2.set_index(['country', 'dataset']) -1) *100
change_plot.columns = [str(x-1) + '-' + str(x) for x in change_plot.columns]

fig, axs = plt.subplots(figsize=(25, 15), nrows=len(change_plot.columns), sharex=True)
for i in range(len(change_plot.columns)):
    temp = change_plot.reset_index()
    sns.barplot(ax=axs[i], data=temp, y=change_plot.columns[i], x='country', hue='dataset')
    axs[i].legend(bbox_to_anchor=(1,1))
plt.xticks(rotation=90)
plt.savefig(plot_filepath + 'barplot_CHANGE.png', dpi=200, bbox_inches='tight')


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
plt.savefig(plot_filepath + 'boxplot_country_difference.png', dpi=200, bbox_inches='tight')
plt.show()

summary_diff = order[['Abs_diff']]
summary_diff['Under'] = 'Other'
for i in [50, 20, 10, 5, 1]:
    summary_diff.loc[summary_diff['Abs_diff'] <= i/100, 'Under'] = str(i) + '%'

sns.barplot(data=change6.reset_index(), x='country', y='pct_same', hue='dataset'); plt.xticks(rotation=90); plt.show()

change_country = change6.mean(axis=0, level='country').sort_values('pct_same', ascending=False)
change_data = change6.mean(axis=0, level='dataset').sort_values('pct_same', ascending=False)

fig, ax = plt.subplots(figsize=(15, 5))
sns.boxplot(ax=ax, data=change6.swaplevel(axis=0).loc[change_country.index.tolist()].reset_index(), x='country', y='pct_same'); plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1));
plt.savefig(plot_filepath + 'boxplot_country_pctsame_bycountry.png', dpi=200, bbox_inches='tight')
plt.show()

sns.boxplot(data=change6.loc[change_data.index.tolist()].reset_index(), x='dataset', y='pct_same'); plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1)); plt.show()
plt.savefig(plot_filepath + 'boxplot_country_pctsame_bydatapair.png', dpi=200, bbox_inches='tight')
plt.show()

sns.scatterplot(data=change6.swaplevel(axis=0).loc[change_country.index.tolist()].reset_index(), x='country', y='pct_same', hue='dataset'); plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1)); plt.show()
plt.savefig(plot_filepath + 'scatterlot_country_pctsame.png', dpi=200, bbox_inches='tight')
