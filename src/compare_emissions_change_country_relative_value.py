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
plot_filepath = wd + 'ESCoE_Project/outputs/compare_all_outputs/plots/'

co2_all = pickle.load(open(emissions_filepath + 'Emissions_aggregated_all.p', 'rb'))

datasets = list(co2_all.keys())
years = list(co2_all[datasets[0]].keys())

data_comb = ['oecd, figaro', 'oecd, exio', 'oecd, gloria', 'figaro, exio', 'figaro, gloria', 'exio, gloria']

def rmspe(x1, x2):
    pct_diff = ((x1/x2) - 1) * 100
    pct_sq = pct_diff **2
    mean_sq = np.mean(pct_sq)
    error = np.sqrt(mean_sq)
    return(error)

###############
## Summarise ##
###############

# Total

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
    
# Imports

summary_im = pd.DataFrame()
for year in years:
    temp = {}
    for item in ['oecd', 'figaro', 'gloria', 'exio']:
        temp[item] = co2_all[item][year]
        for country in temp[item].index.levels[0]:
            temp[item].loc[country, country] = 0
        temp[item] = pd.DataFrame(temp[item].sum(axis=1, level=0).sum(axis=0)).rename(columns={0:item})
        
    temp_all = temp['oecd'].join(temp['figaro']).join(temp['gloria']).join(temp['exio'])
    temp_all['year'] = year
    summary_im = summary_im.append(temp_all.reset_index())
    
summary_im = summary_im.rename(columns={'index':'country'}).set_index(['country', 'year'])

#####################
## Change in trend ##
#####################

# Total

temp = summary.unstack('country').swaplevel(axis=1)
change = pd.DataFrame(columns=['country'])
# Convert to True vs False
for c in temp.columns.levels[0]:
    temp2 = temp[c]
    for comb in data_comb:
        d0 = comb.split(', ')[0]
        d1 = comb.split(', ')[1]
    
        temp3 = pd.DataFrame(index=[0])
        temp3['country'] = c
        temp3[comb] = (rmspe(temp2[d0], temp2[d1]) + rmspe(temp2[d1], temp2[d0]))/2
        
        temp3 = temp3.set_index('country').stack().reset_index().rename(columns={'level_1':'dataset', 0:'RMSPE'})
        
        change = change.append(temp3)
        
change = change.merge(change.groupby('country').mean().reset_index().rename(columns={'RMSPE':'mean'}), on='country').sort_values('mean')

# Imports

temp = summary.unstack('country').swaplevel(axis=1)
change_im = pd.DataFrame(columns=['country'])
# Convert to True vs False
for c in temp.columns.levels[0]:
    temp2 = temp[c]
    for comb in data_comb:
        d0 = comb.split(', ')[0]
        d1 = comb.split(', ')[1]
    
        temp3 = pd.DataFrame(index=[0])
        temp3['country'] = c
        temp3[comb] = (rmspe(temp2[d0], temp2[d1]) + rmspe(temp2[d1], temp2[d0]))/2
        
        temp3 = temp3.set_index('country').stack().reset_index().rename(columns={'level_1':'dataset', 0:'RMSPE'})
        
        change_im = change_im.append(temp3)
        
change_im = change_im.merge(change_im.groupby('country').mean().reset_index().rename(columns={'RMSPE':'mean'}), on='country').sort_values('mean')


###################
## Plot together ##
###################

fs = 16
pal = 'tab10'
c_box = '#000000'
c_vlines = '#B9B9B9'

country_dict = {'United Kingdom':'UK', 'South Korea':'S. Korea', 'Czech Republic':'Czechia', 'United States':'USA', 'Rest of the World':'RoW'}
for item in change['country'].unique():
    if item not in list(country_dict.keys()):
        country_dict[item] = item
data_dict = {'oecd, figaro':'ICIO, Figaro', 'oecd, exio':'Exiobase, ICIO', 'oecd, gloria':'ICIO, Gloria', 
             'figaro, exio':'Exiobase, Figaro', 'figaro, gloria':'Figaro, Gloria', 'exio, gloria':'Exiobase, Gloria'}

fig, axs = plt.subplots(nrows=2, figsize=(20, 10), sharex=True)

plot_data = cp.copy(change)
plot_data['country'] = plot_data['country'].map(country_dict)
plot_data['dataset'] = plot_data['dataset'].map(data_dict)
plot_data['country'] = '                     ' + plot_data['country']
sns.stripplot(ax=axs[0], data=plot_data, x='country', y='RMSPE', hue='dataset', s=8, jitter=0.4, palette=pal); 

plot_data_imports = cp.copy(change_im)
plot_data_imports['country'] = plot_data_imports['country'].map(country_dict)
plot_data_imports['dataset'] = plot_data_imports['dataset'].map(data_dict)
plot_data_imports['country'] = '                     ' + plot_data_imports['country']
sns.stripplot(ax=axs[1], data=plot_data_imports, x='country', y='RMSPE', hue='dataset', s=8, jitter=0.4, palette=pal); 

plt.setp(axs[0].artists, edgecolor=c_box, facecolor='w')
sns.boxplot(ax=axs[0], data=plot_data, x='country', y='RMSPE', color='w', showfliers=False) 
sns.boxplot(ax=axs[1], data=plot_data_imports, x='country', y='RMSPE', color='w', showfliers=False)

axs[0].set_ylabel('Total footprint RMSPE (%)', fontsize=fs); 
axs[1].set_ylabel('Imports RMSPE (%)', fontsize=fs)

axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=90, va='center', fontsize=fs); 
axs[1].xaxis.set_ticks_position('top') # the rest is the same

for i in range(2):
    axs[i].set_ylim(-5, 105)
    axs[i].set_xlabel('')
    axs[i].tick_params(axis='y', labelsize=fs)
    axs[i].legend(loc='upper center', fontsize=fs, ncol=len(plot_data['dataset'].unique()))
    for c in range(len(plot_data['country'].unique())):
        axs[i].axvline(c+0.5, c=c_vlines, linestyle=':')

fig.tight_layout()
plt.savefig(plot_filepath + 'ALL_stripplot_country_RMSPE_bycountry.png', dpi=200, bbox_inches='tight')
plt.show()
