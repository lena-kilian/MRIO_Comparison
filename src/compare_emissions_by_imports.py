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
import math
import matplotlib as mpl

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

prop_im = summary_im / summary
prop_im = prop_im.mean(axis=0, level=0)
prop_im['prop_im_mean'] = prop_im.mean(1)
prop_im['percentage_imported'] = [math.floor(x * 100) for x in prop_im['prop_im_mean']]

pal = 'RdBu' # 'viridis' # 'Spectral' 
cols = sns.color_palette(pal, as_cmap=True); cols2 = cols(np.linspace(0, 1, 100))
prop_im['percentage_imported_c']  = [cols2[x] for x in prop_im['percentage_imported']]

prop_im['percentage_imported'] = prop_im['prop_im_mean'] * 100

steps = 20
prop_im['percentage_imported_steps'] = 'NA'
for i in range(0, 100, steps):
    prop_im.loc[(prop_im['percentage_imported'] >= i) & (prop_im['percentage_imported'] <= i+20), 
                'percentage_imported_steps'] = str(i) + ' - ' + str(i+steps)
    

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
change = change.set_index('country').join(prop_im[['percentage_imported', 'percentage_imported_c', 'percentage_imported_steps']])\
    .reset_index().sort_values('percentage_imported')

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
        
change_im = change_im.merge(change_im.groupby('country').mean().reset_index().rename(columns={'RMSPE':'mean'}), on='country').sort_values('mean')
change_im = change_im.set_index('country').join(prop_im[['percentage_imported', 'percentage_imported_c', 'percentage_imported_steps']])\
    .reset_index().sort_values('percentage_imported')

###################
## Plot together ##
###################

fs = 16

country_dict = {'United Kingdom':'UK', 'South Korea':'S. Korea', 'Czech Republic':'Czechia', 'United States':'USA', 'Rest of the World':'RoW'}
for item in change['country'].unique():
    if item not in list(country_dict.keys()):
        country_dict[item] = item
        
data_dict = {'oecd, figaro':'ICIO, Figaro', 'oecd, exio':'Exiobase, ICIO', 'oecd, gloria':'ICIO, Gloria', 
             'figaro, exio':'Exiobase, Figaro', 'figaro, gloria':'Figaro, Gloria', 'exio, gloria':'Exiobase, Gloria'}

change['dataset'] = change['dataset'].map(data_dict)
change_im['dataset'] = change_im['dataset'].map(data_dict)

# Plot with data on x
fig, axs = plt.subplots(figsize=(15, 10), nrows=2, sharex=True)
plot_data_im = change_im.groupby('country').describe()['RMSPE'][['mean', 'std']].join(prop_im)\
    .reset_index().sort_values('percentage_imported')
plot_data = change.groupby('country').describe()['RMSPE'][['mean', 'std']].join(prop_im)\
    .loc[plot_data_im['country']].reset_index()
plot_data['country'] = '                     ' + plot_data['country'].map(country_dict)
plot_data_im['country'] = '                     ' + plot_data_im['country'].map(country_dict)
sns.barplot(ax=axs[0], data=plot_data, y='mean', x='country', palette=plot_data['percentage_imported_c'], edgecolor='k')
sns.barplot(ax=axs[1], data=plot_data_im, y='mean', x='country', palette=plot_data_im['percentage_imported_c'], edgecolor='k')

axs[0].set_ylabel('Total footprint RMSPE (%)', fontsize=fs); 
axs[1].set_ylabel('Imports RMSPE (%)', fontsize=fs); 

axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=90, va='center', fontsize=fs); 
axs[1].xaxis.set_ticks_position('top') # the rest is the same

axs[0].set_ylim(0, 43); 
axs[1].set_ylim(0, 260); 

for i in range(2):
    axs[i].set_xlabel(''); 
    for j in range(len(axs[i].get_yticklabels())):
        axs[i].axhline(axs[i].get_yticks()[j], c='k', linestyle=':')
    axs[i].tick_params(axis='x', labelsize=fs, rotation=90)
    axs[i].tick_params(axis='y', labelsize=fs)
    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 100), cmap=pal),
                 ax=axs[i], orientation='vertical', label='Proportion of imported emissions (%)')

fig.tight_layout()
plt.savefig(plot_filepath + 'ALL_barplot_RMSPE_byimports.png', dpi=200, bbox_inches='tight')
plt.show()


corr = change

# Plot with data on x
fig, axs = plt.subplots(figsize=(15, 10), nrows=2, sharex=True)
plot_data_im = change_im.groupby('country').describe()['RMSPE'][['mean', 'std']].join(prop_im)\
    .reset_index().sort_values('percentage_imported')
plot_data = change.groupby('country').describe()['RMSPE'][['mean', 'std']].join(prop_im)\
    .loc[plot_data_im['country']].reset_index()
plot_data['country'] = '                     ' + plot_data['country'].map(country_dict)
plot_data_im['country'] = '                     ' + plot_data_im['country'].map(country_dict)
sns.barplot(ax=axs[0], data=plot_data, y='mean', x='country', palette=plot_data['percentage_imported_c'], edgecolor='k')
sns.barplot(ax=axs[1], data=plot_data_im, y='mean', x='country', palette=plot_data_im['percentage_imported_c'], edgecolor='k')

axs[0].set_ylabel('Total footprint RMSPE (%)', fontsize=fs); 
axs[1].set_ylabel('Imports RMSPE (%)', fontsize=fs); 

axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=90, va='center', fontsize=fs); 
axs[1].xaxis.set_ticks_position('top') # the rest is the same

for i in range(2):
    axs[i].set_ylim(0, 100); 
    axs[i].set_xlabel(''); 
    for j in range(len(axs[i].get_yticklabels())):
        axs[i].axhline(axs[i].get_yticks()[j], c='k', linestyle=':')
    axs[i].tick_params(axis='x', labelsize=fs, rotation=90)
    axs[i].tick_params(axis='y', labelsize=fs)
    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 100), cmap=pal),
                 ax=axs[i], orientation='vertical', label='Proportion of imported emissions (%)')

fig.tight_layout()
plt.savefig(plot_filepath + 'ALL_barplot_RMSPE_byimportsZOOM.png', dpi=200, bbox_inches='tight')
plt.show()

# Plot with data on x

fig, axs = plt.subplots(figsize=(15, 10), nrows=2)
sns.scatterplot(ax=axs[0], data=change, hue='dataset', y='RMSPE', x='percentage_imported')
sns.scatterplot(ax=axs[1], data=change_im, hue='dataset', y='RMSPE', x='percentage_imported')

axs[0].set_ylabel('Total footprint RMSPE (%)', fontsize=fs); 
axs[1].set_ylabel('Imports RMSPE (%)', fontsize=fs); 

for i in range(2):
    axs[i].set_xlabel('Emissions imported (%)', fontsize=fs); 
    axs[i].tick_params(axis='x', labelsize=fs)
    axs[i].tick_params(axis='y', labelsize=fs)
    axs[i].legend(fontsize=fs)

fig.tight_layout()
plt.savefig(plot_filepath + 'ALL_scatterplot_RMSPE_byimports.png', dpi=200, bbox_inches='tight')
plt.show()



sns.lmplot(data=change, hue='dataset', y='RMSPE', x='percentage_imported', ci=None, size=7.5, legend=False)
#plt.ylim(0, 100); plt.xlim(0, 100)
plt.ylabel('Total footprint RMSPE (%)', fontsize=fs); 
plt.xlabel('Emissions imported (%)', fontsize=fs); 
plt.tick_params(axis='x', labelsize=fs)
plt.tick_params(axis='y', labelsize=fs)
plt.legend(fontsize=fs)
plt.savefig(plot_filepath + 'ALL_lmplot_RMSPE_total_bydata.png', dpi=200, bbox_inches='tight')
plt.show()

sns.lmplot(data=change_im, hue='dataset', y='RMSPE', x='percentage_imported', ci=None, size=7.5, legend=False)
#plt.ylim(0, 100); plt.xlim(0, 100)
plt.ylabel('Imports RMSPE (%)', fontsize=fs); 
plt.xlabel('Emissions imported (%)', fontsize=fs); 
plt.tick_params(axis='x', labelsize=fs)
plt.tick_params(axis='y', labelsize=fs)
plt.legend(fontsize=fs)
plt.savefig(plot_filepath + 'ALL_lmplot_RMSPE_imports_bydata.png', dpi=200, bbox_inches='tight')
plt.show()