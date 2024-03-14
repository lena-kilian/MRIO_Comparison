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

temp = summary.unstack(level=1).stack(level=0)
change = temp[years[1:]]
for year in years[1:]:
    change[year] = temp[year] / temp[year - 1]
    
change = change.unstack(level=1).stack(level=0)
# Convert to True vs False
for comb in data_comb:
    d0 = comb.split(', ')[0]
    d1 = comb.split(', ')[1]
    
    change[comb] = False
    change.loc[((change[d0]>1) & (change[d1]>1) | (change[d0]<1) & (change[d1]<1) | (change[d0]==1) & (change[d1]==1)), comb] = True

change = change[data_comb].stack().reset_index().rename(columns={'level_2':'dataset', 0:'Same_direction'})
change['count'] = 1
change = change.set_index(['country', 'year', 'dataset', 'Same_direction']).unstack('Same_direction').droplevel(axis=1, level=0).fillna(0).sum(axis=0, level=[0, 2])

change['pct_same'] = change[True] / (change[True] + change[False])*100

change_country = change.mean(axis=0, level='country').sort_values('pct_same', ascending=False)
change_country['rank'] = list(range(1, len(change_country) + 1))

results = change.reset_index().merge(change_country[['rank']], on='country')


# Imports

temp = summary_im.unstack(level=1).stack(level=0)
change_im = temp[years[1:]]
for year in years[1:]:
    change_im[year] = temp[year] / temp[year - 1]
    
change_im = change_im.unstack(level=1).stack(level=0)
# Convert to True vs False
for comb in data_comb:
    d0 = comb.split(', ')[0]
    d1 = comb.split(', ')[1]
    
    change_im[comb] = False
    change_im.loc[((change_im[d0]>1) & (change_im[d1]>1) | (change_im[d0]<1) & (change_im[d1]<1) | (change_im[d0]==1) & (change_im[d1]==1)), comb] = True

change_im = change_im[data_comb].stack().reset_index().rename(columns={'level_2':'dataset', 0:'Same_direction'})
change_im['count'] = 1
change_im = change_im.set_index(['country', 'year', 'dataset', 'Same_direction']).unstack('Same_direction').droplevel(axis=1, level=0).fillna(0).sum(axis=0, level=[0, 2])

change_im['pct_same'] = change_im[True] / (change_im[True] + change_im[False])*100

change_im_country = change_im.mean(axis=0, level='country').sort_values('pct_same', ascending=False)
change_im_country['rank'] = list(range(1, len(change_im_country) + 1))

results_im = change_im.reset_index().merge(change_im_country[['rank']], on='country')

###################
## Plot together ##
###################

fs = 16
pal = 'tab10'
c_box = '#000000'
c_vlines = '#B9B9B9'

country_dict = {'United Kingdom':'UK', 'South Korea':'S. Korea', 'Czech Republic':'Czechia', 'United States':'USA', 'Rest of the World':'RoW'}
data_dict = {'oecd, figaro':'ICIO, Figaro', 'oecd, exio':'Exiobase, ICIO', 'oecd, gloria':'ICIO, Gloria', 
             'figaro, exio':'Exiobase, Figaro', 'figaro, gloria':'Figaro, Gloria', 'exio, gloria':'Exiobase, Gloria'}

fig, axs = plt.subplots(nrows=2, figsize=(20, 10), sharex=True)

plot_data = change.loc[change_country.index.tolist()].rename(index=country_dict).swaplevel(axis=0).rename(index=data_dict).reset_index()
plot_data['country'] = '                     ' + plot_data['country']
sns.stripplot(ax=axs[0], data=plot_data,
              x='country', y='pct_same', hue='dataset', s=8, jitter=0.4, palette=pal); 

plot_data_imports = change_im.loc[change_country.index.tolist()].rename(index=country_dict).swaplevel(axis=0).rename(index=data_dict).reset_index()
plot_data_imports['country'] = '                     ' + plot_data_imports['country']
sns.stripplot(ax=axs[1], data=plot_data_imports, 
              x='country', y='pct_same', hue='dataset', s=8, jitter=0.4, palette=pal); 

plt.setp(axs[0].artists, edgecolor=c_box, facecolor='w')
sns.boxplot(ax=axs[0], data=plot_data, x='country', y='pct_same', color='w', showfliers=False) 
sns.boxplot(ax=axs[1], data=plot_data_imports, x='country', y='pct_same', color='w', showfliers=False)

axs[0].set_ylabel('Total footprint similarity (%)', fontsize=fs); 
axs[1].set_ylabel('Imports similarity (%)', fontsize=fs)

axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=90, va='center', fontsize=fs); 
axs[1].xaxis.set_ticks_position('top') # the rest is the same

for i in range(2):
    axs[i].set_ylim(-5, 105)
    axs[i].set_xlabel('')
    axs[i].tick_params(axis='y', labelsize=fs)
    axs[i].legend(loc='lower center', fontsize=fs, ncol=len(plot_data['dataset'].unique()))
    for c in range(len(plot_data['country'].unique())):
        axs[i].axvline(c+0.5, c=c_vlines, linestyle=':')

fig.tight_layout()
plt.savefig(plot_filepath + 'ALL_stripplot_country_pctsame_bycountry.png', dpi=200, bbox_inches='tight')
plt.show()


# Plot with data on x

plot_data2 = plot_data.set_index(['country', 'dataset'])[['pct_same']].join(plot_data_imports.set_index(['country', 'dataset'])[['pct_same']], rsuffix='_imports')\
    .stack().reset_index().rename(columns={0:'pct_same'})
temp = plot_data2.loc[plot_data2['Same_direction'] == 'pct_same'].groupby(['dataset']).mean().rename(columns={'pct_same':'mean'}).reset_index()
plot_data2 = plot_data2.merge(temp, on='dataset').sort_values('mean', ascending=False)
plot_data2['Type'] = plot_data2['Same_direction'].map({'pct_same':'Total', 'pct_same_imports':'Imports'})

fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(ax=ax, data=plot_data2, x='dataset', y='pct_same', hue='Type', showfliers=False)
sns.stripplot(ax=ax, data=plot_data2, x='dataset', y='pct_same', hue='Type', dodge=True, palette='dark', alpha=0.6, s=7.5)

ax.set_ylabel('Footprint similarity (%)', fontsize=fs); 

ax.set_xticklabels(ax.get_xticklabels(), fontsize=fs); 

for i in range(2):
    ax.set_ylim(0, 101)
    ax.set_xlabel('')
    ax.tick_params(axis='y', labelsize=fs)
    ax.legend(loc='lower center', fontsize=fs, ncol=len(plot_data['dataset'].unique()))

fig.tight_layout()
plt.savefig(plot_filepath + 'ALL_boxplot_country_pctsame_bydata.png', dpi=200, bbox_inches='tight')
plt.show()
