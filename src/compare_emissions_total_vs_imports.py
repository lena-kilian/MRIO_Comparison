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


country_dict = {'United Kingdom':'UK', 'Czech Republic':'Czechia', 'United States':'USA', 'Rest of the World':'RoW'}
data_dict = {'oecd':'ICIO', 'exio':'Exiobase', 'gloria':'Gloria', 'figaro':'Figaro'}

data_comb = ['oecd, figaro', 'oecd, exio', 'oecd, gloria', 'figaro, exio', 'figaro, gloria', 'exio, gloria']

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

###################
## Plot together ##
###################

fs = 16
pal = 'tab10'
c_box = '#000000'
c_vlines = '#B9B9B9'

mean_co2 = summary.mean(axis=0, level='country')
mean_co2_im = summary_im.mean(axis=0, level='country')
percent_im = pd.DataFrame((mean_co2_im / mean_co2 * 100).stack()).reset_index()
percent_im.columns = ['country', 'dataset', 'pct_im']

order = mean_co2.stack().mean(axis=0, level='country').sort_values(0, ascending=False).index.tolist()

percent_im['Dataset'] = percent_im['dataset'].map(data_dict)
percent_im = percent_im.sort_values('Dataset').set_index('country').loc[order].rename(index=country_dict).reset_index()

plot_data = mean_co2.stack().reset_index(); plot_data['Type'] = 'Total'
temp = mean_co2_im.stack().reset_index(); temp['Type'] = 'Imports'
plot_data = plot_data.append(temp).drop_duplicates()
plot_data.columns = ['country', 'dataset', 'CO2', 'Type']
plot_data.index = list(range(len(plot_data)))

plot_data['Dataset'] = plot_data['dataset'].map(data_dict)
plot_data = plot_data.sort_values('Dataset').set_index('country').loc[order].rename(index=country_dict).reset_index()


## Linplots

fig, ax1 = plt.subplots(nrows=1, figsize=(20, 5))

temp = cp.copy(plot_data)
temp.loc[temp['Type'] == 'Imports', 'CO2'] = 0
temp['Linetype'] = temp['Type'].map({'Total':'Total emissions', 'Imports':'Proportion imported (%)'})
temp = temp.loc[(temp['Type'] == 'Total') | (temp['country'] == 'India')]


sns.lineplot(ax=ax1, data=temp, x='country', y='CO2', hue='Dataset', style='Linetype')
#plt.yscale('log')

axs0_2 = ax1.twinx()
sns.lineplot(ax=axs0_2, data=percent_im, y='pct_im', x='country', hue='Dataset', linestyle='--')
axs0_2.tick_params(axis='y', labelsize=fs)
axs0_2.set_ylabel('Emisions imported (%)', fontsize=fs); 

ax1.set_ylabel('Footprint (CO2)', fontsize=fs); 

ax1.set_xlabel('')
ax1.tick_params(axis='y', labelsize=fs)
ax1.tick_params(axis='x', labelsize=fs, rotation=90)
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), fontsize=fs, ncol=len(plot_data['Dataset'].unique())+4)
axs0_2.legend(bbox_to_anchor=(1.12, 1))
for c in range(len(plot_data['country'].unique())):
    ax1.axvline(c+0.5, c=c_vlines, linestyle=':')
    

fig.tight_layout()
plt.savefig(plot_filepath + 'Lineplot_overview_bycountry.png', dpi=200, bbox_inches='tight')
plt.show()
