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

data_comb = ['ICIO, Figaro', 'Exiobase, ICIO', 'ICIO, Gloria', 'Exiobase, Figaro', 'Figaro, Gloria', 'Exiobase, Gloria']

def calc_rmspe(x1, x2):
    pct_diff = ((x1/x2) - 1) * 100
    pct_sq = pct_diff **2
    mean_sq = np.mean(pct_sq)
    error = np.sqrt(mean_sq)
    return(error)

def calc_rmse(x1, x2):
    diff = x1-x2
    diff_sq = diff**2
    mean_sq = np.mean(diff_sq)
    error = np.sqrt(mean_sq)
    return(error)

country_dict = {'United Kingdom':'UK', 'Czech Republic':'Czechia', 'United States':'USA', 'Rest of the World':'RoW'}
data_dict = {'oecd':'ICIO', 'exio':'Exiobase', 'gloria':'Gloria', 'figaro':'Figaro'}

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
    # merge all
    temp = temp_oecd.join(temp_figaro).join(temp_gloria).join(temp_exio) 
    temp['year'] = year
    summary = summary.append(temp.reset_index())
summary = summary.rename(columns={'index':'country'}).set_index(['country', 'year']).rename(index=country_dict).rename(columns=data_dict)
    
# Imports

summary_im = pd.DataFrame()
for year in years:
    temp = {}
    for item in ['oecd', 'figaro', 'gloria', 'exio']:
        temp[item] = co2_all[item][year]
        for country in temp[item].index.levels[0]:
            temp[item].loc[country, country] = 0
        temp[item] = pd.DataFrame(temp[item].sum(axis=1, level=0).sum(axis=0)).rename(columns={0:item})
    # merge all
    temp_all = temp['oecd'].join(temp['figaro']).join(temp['gloria']).join(temp['exio'])
    temp_all['year'] = year
    summary_im = summary_im.append(temp_all.reset_index())
summary_im = summary_im.rename(columns={'index':'country'}).set_index(['country', 'year']).rename(index=country_dict).rename(columns=data_dict)

# Get means

mean_co2 = {'Total' : pd.DataFrame(summary.mean(axis=0, level='country').mean(axis=1)).rename(columns={0:'mean_co2'}), 
            'Imports' : pd.DataFrame(summary_im.mean(axis=0, level='country').mean(axis=1)).rename(columns={0:'mean_co2'})}

prop_im = (summary_im / summary * 100).mean(axis=0, level='country').mean(axis=1).reset_index().rename(columns={0:'prop_imported'})

#############################
## Change in trend - RMSPE ##
#############################

# Total

temp = summary.unstack('country').swaplevel(axis=1)
data_rmspe = pd.DataFrame(columns=['country'])
# Convert to True vs False
for c in temp.columns.levels[0]:
    temp2 = temp[c]
    for comb in data_comb:
        d0 = comb.split(', ')[0]
        d1 = comb.split(', ')[1]
        temp3 = pd.DataFrame(index=[0])
        temp3['country'] = c
        temp3[comb] = (calc_rmspe(temp2[d0], temp2[d1]) + calc_rmspe(temp2[d1], temp2[d0]))/2
        temp3 = temp3.set_index('country').stack().reset_index().rename(columns={'level_1':'dataset', 0:'RMSPE'})
        data_rmspe = data_rmspe.append(temp3)
data_rmspe = data_rmspe.merge(data_rmspe.groupby('country').mean().reset_index().rename(columns={'RMSPE':'mean'}), on='country').sort_values(['mean', 'dataset'])
data_rmspe = data_rmspe.merge(prop_im, on ='country')

# Imports

temp = summary_im.unstack('country').swaplevel(axis=1)
data_rmspe_im = pd.DataFrame(columns=['country'])
# Convert to True vs False
for c in temp.columns.levels[0]:
    temp2 = temp[c]
    for comb in data_comb:
        d0 = comb.split(', ')[0]
        d1 = comb.split(', ')[1]
        temp3 = pd.DataFrame(index=[0])
        temp3['country'] = c
        temp3[comb] = (calc_rmspe(temp2[d0], temp2[d1]) + calc_rmspe(temp2[d1], temp2[d0]))/2
        temp3 = temp3.set_index('country').stack().reset_index().rename(columns={'level_1':'dataset', 0:'RMSPE'})
        data_rmspe_im = data_rmspe_im.append(temp3)
data_rmspe_im = data_rmspe_im.merge(data_rmspe_im.groupby('country').mean().reset_index().rename(columns={'RMSPE':'mean'}), on='country')
data_rmspe_im = data_rmspe_im.merge(prop_im, on ='country')

# Combine all

data_rmspe = {'Total':data_rmspe, 'Imports':data_rmspe_im}

#################################
## Change in trend - Direction ##
#################################

# Total

temp = summary.unstack(level=1).stack(level=0)
data_direction = temp[years[1:]]
for year in years[1:]:
    data_direction[year] = temp[year] / temp[year - 1]
data_direction = data_direction.unstack(level=1).stack(level=0)
# Convert to True vs False
for comb in data_comb:
    d0 = comb.split(', ')[0]
    d1 = comb.split(', ')[1]
    data_direction[comb] = False
    data_direction.loc[((data_direction[d0]>1) & (data_direction[d1]>1) | (data_direction[d0]<1) & (data_direction[d1]<1) | 
                        (data_direction[d0]==1) & (data_direction[d1]==1)), comb] = True
data_direction = data_direction[data_comb].stack().reset_index().rename(columns={'level_2':'dataset', 0:'Same_direction'})
data_direction['count'] = 1
data_direction = data_direction.set_index(['country', 'year', 'dataset', 'Same_direction']).unstack('Same_direction')\
    .droplevel(axis=1, level=0).fillna(0).sum(axis=0, level=[0, 2])
data_direction['pct_same'] = data_direction[True] / (data_direction[True] + data_direction[False])*100
data_direction = data_direction.reset_index().merge(prop_im, on='country')

# Imports

temp = summary_im.unstack(level=1).stack(level=0)
data_direction_im = temp[years[1:]]
for year in years[1:]:
    data_direction_im[year] = temp[year] / temp[year - 1]
data_direction_im = data_direction_im.unstack(level=1).stack(level=0)
# Convert to True vs False
for comb in data_comb:
    d0 = comb.split(', ')[0]
    d1 = comb.split(', ')[1]
    data_direction_im[comb] = False
    data_direction_im.loc[((data_direction_im[d0]>1) & (data_direction_im[d1]>1) | (data_direction_im[d0]<1) & (data_direction_im[d1]<1) | 
                           (data_direction_im[d0]==1) & (data_direction_im[d1]==1)), comb] = True
data_direction_im = data_direction_im[data_comb].stack().reset_index().rename(columns={'level_2':'dataset', 0:'Same_direction'})
data_direction_im['count'] = 1
data_direction_im = data_direction_im.set_index(['country', 'year', 'dataset', 'Same_direction']).unstack('Same_direction')\
    .droplevel(axis=1, level=0).fillna(0).sum(axis=0, level=[0, 2])
data_direction_im['pct_same'] = data_direction_im[True] / (data_direction_im[True] + data_direction_im[False])*100
data_direction_im = data_direction_im.reset_index().merge(prop_im, on='country')

# Combine all

data_direction = {'Total':data_direction, 'Imports':data_direction_im}

###################
## Plot together ##
###################

fs = 16

corr = {}
for item in ['Total', 'Imports']:
    
    data_direction[item]['Dataset'] = data_direction[item]['dataset'] 
    data_direction[item] = data_direction[item].sort_values('Dataset')
    sns.lmplot(data=data_direction[item], hue='Dataset', y='pct_same', x='prop_imported', ci=None, size=7.5, legend=False)
    #plt.ylim(0, 50); #plt.xlim(0, 100)
    plt.ylabel('Similar direction (%)', fontsize=fs); 
    plt.xlabel('Emissions imported (%)', fontsize=fs); 
    plt.title(item, fontsize=fs); 
    plt.tick_params(axis='x', labelsize=fs)
    plt.tick_params(axis='y', labelsize=fs)
    plt.legend(fontsize=fs)
    plt.savefig(plot_filepath + 'lmplot_Direction_' + item + '_bydata_COLOUR.png', dpi=200, bbox_inches='tight')
    plt.show()
     
    data_rmspe[item]['Dataset'] = data_rmspe[item]['dataset'] 
    data_rmspe[item] = data_rmspe[item].sort_values('Dataset')
    sns.lmplot(data=data_rmspe[item], hue='dataset', y='RMSPE', x='prop_imported', ci=None, size=7.5, legend=False)
    #plt.ylim(0, 50); #plt.xlim(0, 100)
    plt.ylabel('RMSPE (%)', fontsize=fs); 
    plt.xlabel('Emissions imported (%)', fontsize=fs); 
    plt.title(item, fontsize=fs); 
    plt.tick_params(axis='x', labelsize=fs)
    plt.tick_params(axis='y', labelsize=fs)
    plt.legend(fontsize=fs)
    plt.savefig(plot_filepath + 'lmplot_RMSPE_' + item + '_bydata_COLOUR.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    corr[item] = pd.DataFrame(data_direction[item].groupby(['dataset']).corr().swaplevel(axis=0).loc['pct_same', 'prop_imported']).join(
        pd.DataFrame(pd.DataFrame(data_rmspe[item].groupby(['dataset']).corr().swaplevel(axis=0).loc['RMSPE', 'prop_imported'])),
        lsuffix = '_direction', rsuffix = '_rmspe')

