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
        
change = change.merge(change.groupby('country').mean().reset_index().rename(columns={'RMSPE':'mean'}), on='country').sort_values(['mean', 'dataset'])

# Imports

temp = summary_im.unstack('country').swaplevel(axis=1)
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
        
change_im = change_im.merge(change_im.groupby('country').mean().reset_index().rename(columns={'RMSPE':'mean'}), on='country')

###################
## Plot together ##
###################

# Plot with country on x

mean_co2 = pd.DataFrame(summary.mean(axis=0, level='country').mean(axis=1)).rename(columns={0:'mean_co2'})
mean_co2_im = pd.DataFrame(summary_im.mean(axis=0, level='country').mean(axis=1)).rename(columns={0:'mean_co2'})

order = mean_co2.sort_values('mean_co2', ascending=False).index.tolist()

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

plot_data = cp.copy(change).merge(mean_co2, on='country').sort_values(['dataset']).set_index('country').loc[order].rename(index=country_dict).reset_index()
plot_data['dataset'] = plot_data['dataset'].map(data_dict)
plot_data['country'] = '                     ' + plot_data['country']
sns.stripplot(ax=axs[0], data=plot_data, x='country', y='RMSPE', hue='dataset', s=8, jitter=0.4, palette=pal); 

#axs0_2 = axs[0].twinx()
#sns.lineplot(ax=axs0_2, data=plot_data, y='mean_co2', x='country', color='k')
#axs0_2.tick_params(axis='y', labelsize=fs)
#axs0_2.set_ylabel('Footprint (CO2)', fontsize=fs); 


plot_data_imports = cp.copy(change_im).merge(mean_co2_im, on='country').sort_values(['dataset']).set_index('country').loc[order].rename(index=country_dict).reset_index()
plot_data_imports['dataset'] = plot_data_imports['dataset'].map(data_dict)
plot_data_imports['country'] = '                     ' + plot_data_imports['country']
plot_data_imports = plot_data_imports.sort_values(['dataset']).set_index('country').loc[plot_data['country'].unique()].reset_index()
sns.stripplot(ax=axs[1], data=plot_data_imports, x='country', y='RMSPE', hue='dataset', s=8, jitter=0.4, palette=pal); 

#axs1_2 = axs[1].twinx()
#sns.lineplot(ax=axs1_2, data=plot_data_imports, y='mean_co2', x='country', color='k')
#axs1_2.tick_params(axis='y', labelsize=fs)
#axs1_2.set_ylabel('Footprint (CO2)', fontsize=fs); 


plt.setp(axs[0].artists, edgecolor=c_box, facecolor='w')
#sns.boxplot(ax=axs[0], data=plot_data, x='country', y='RMSPE', color='w', showfliers=False) 
#sns.boxplot(ax=axs[1], data=plot_data_imports, x='country', y='RMSPE', color='w', showfliers=False)

axs[0].set_ylabel('Total footprint RMSPE (%)', fontsize=fs); 
axs[1].set_ylabel('Imports RMSPE (%)', fontsize=fs)

axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=90, va='center', fontsize=fs); 
axs[1].xaxis.set_ticks_position('top') # the rest is the same

for i in range(2):
    #axs[i].set_ylim(-5, 105)
    axs[i].set_xlabel('')
    axs[i].tick_params(axis='y', labelsize=fs)
    axs[i].legend(loc='upper center', fontsize=fs, ncol=len(plot_data['dataset'].unique()))
    for c in range(len(plot_data['country'].unique())):
        axs[i].axvline(c+0.5, c=c_vlines, linestyle=':')

fig.tight_layout()
plt.savefig(plot_filepath + 'ALL_stripplot_country_RMSPE_bycountry.png', dpi=200, bbox_inches='tight')
plt.show()


# Plot with data on x

plot_data2 = plot_data.set_index(['country', 'dataset'])[['RMSPE']].join(plot_data_imports.set_index(['country', 'dataset'])[['RMSPE']], rsuffix='_imports')\
    .stack().reset_index().rename(columns={0:'RMSPE'})
temp = plot_data2.loc[plot_data2['level_2'] == 'RMSPE'].groupby(['dataset']).mean().rename(columns={'RMSPE':'mean'}).reset_index()
plot_data2 = plot_data2.merge(temp, on='dataset').sort_values('mean')
plot_data2['Type'] = plot_data2['level_2'].map({'RMSPE':'Total', 'RMSPE_imports':'Imports'})

fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(ax=ax, data=plot_data2, x='dataset', y='RMSPE', hue='Type', showfliers=False)
sns.stripplot(ax=ax, data=plot_data2, x='dataset', y='RMSPE', hue='Type', dodge=True, palette='dark', alpha=0.6, s=7.5)

ax.set_ylabel('Footprint RMSPE (%)', fontsize=fs); 

ax.set_xticklabels(ax.get_xticklabels(), fontsize=fs); 

for i in range(2):
    ax.set_ylim(0, 150)
    ax.set_xlabel('')
    ax.tick_params(axis='y', labelsize=fs)
    ax.legend(loc='upper center', fontsize=fs, ncol=len(plot_data['dataset'].unique()))

fig.tight_layout()
plt.savefig(plot_filepath + 'ALL_boxplot_country_RMSPE_bydata.png', dpi=200, bbox_inches='tight')
plt.show()
