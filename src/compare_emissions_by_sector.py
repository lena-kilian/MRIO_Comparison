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
    temp_oecd = pd.DataFrame(co2_all['oecd'][year].sum(axis=1, level=0).sum(axis=0, level=1).stack()).rename(columns={0:'oecd'})
    temp_figaro = pd.DataFrame(co2_all['figaro'][year].sum(axis=1, level=0).sum(axis=0, level=1).stack()).rename(columns={0:'figaro'})
    temp_gloria = pd.DataFrame(co2_all['gloria'][year].sum(axis=1, level=0).sum(axis=0, level=1).stack()).rename(columns={0:'gloria'})
    temp_exio = pd.DataFrame(co2_all['exio'][year].sum(axis=1, level=0).sum(axis=0, level=1).stack()).rename(columns={0:'exio'})
    
    temp = temp_oecd.join(temp_figaro).join(temp_gloria).join(temp_exio)
    
    temp['year'] = year
    
    summary = summary.append(temp.reset_index())
    
summary = summary.rename(columns={'level_0':'sector', 'level_1':'country'}).set_index(['country', 'sector', 'year'])

sector_co2 = pd.DataFrame(summary.sum(axis=0, level='sector').mean(axis=1)).rename(columns={0:'mean_co2'}).mean(axis=0, level=0).sort_values('mean_co2', ascending=False)
sector_co2['mean_co2_pct'] = sector_co2['mean_co2'] / sector_co2['mean_co2'].sum() * 100
sector_co2['cum_pct'] = sector_co2['mean_co2_pct'].cumsum()

l = len(sector_co2[sector_co2['cum_pct'] < 80].index)
top_sectors = sector_co2.iloc[:l+1, :].index.tolist()

summary = summary.sum(axis=0, level=['country', 'year'])

# by sector

summary_sector = {}
prop_sector = {}

for sector in top_sectors:
    summary_sector[sector] = pd.DataFrame()
    for year in years:
        temp = {}
        for item in ['oecd', 'figaro', 'gloria', 'exio']:
            temp[item] = pd.DataFrame(co2_all[item][year].sum(axis=0, level=1).sum(axis=1, level=0).loc[sector]).rename(columns={sector:item})

        temp_all = temp['oecd'].join(temp['figaro']).join(temp['gloria']).join(temp['exio'])
        temp_all['year'] = year
        summary_sector[sector] = summary_sector[sector].append(temp_all.reset_index())
        
    summary_sector[sector] = summary_sector[sector].rename(columns={'index':'country'}).set_index(['country', 'year'])

    prop_sector[sector] = summary_sector[sector] / summary
    prop_sector[sector] = prop_sector[sector].mean(axis=0, level=0)
    prop_sector[sector]['prop_sector[sector]_mean'] = prop_sector[sector].mean(1)
    prop_sector[sector]['percentage_sector'] = [math.floor(x * 100) for x in prop_sector[sector]['prop_sector[sector]_mean']]
    
    pal = 'RdBu' # 'viridis' # 'Spectral' 
    cols = sns.color_palette(pal, as_cmap=True); cols2 = cols(np.linspace(0, 1, 100))
    prop_sector[sector]['percentage_sector_c']  = [cols2[x] for x in prop_sector[sector]['percentage_sector']]
    
    prop_sector[sector]['percentage_sector'] = prop_sector[sector]['prop_sector[sector]_mean'] * 100
    
    steps = 20
    prop_sector[sector]['percentage_sector_steps'] = 'NA'
    for i in range(0, 100, steps):
        prop_sector[sector].loc[(prop_sector[sector]['percentage_sector'] >= i) & (prop_sector[sector]['percentage_sector'] <= i+20), 
                    'percentage_sector_steps'] = str(i) + ' - ' + str(i+steps)
    

#####################
## Change in trend ##
#####################

change_sector = {}

for sector in top_sectors:
    temp = summary_sector[sector].unstack('country').swaplevel(axis=1)
    change_sector[sector] = pd.DataFrame(columns=['country'])
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
            
            change_sector[sector] = change_sector[sector].append(temp3)
            
    change_sector[sector] = change_sector[sector].merge(change_sector[sector].groupby('country').mean().reset_index().rename(columns={'RMSPE':'mean'}), on='country').sort_values('mean')
    change_sector[sector] = change_sector[sector].set_index('country').join(prop_sector[sector][['percentage_sector', 'percentage_sector_c', 'percentage_sector_steps']])\
        .reset_index().sort_values('percentage_sector')

###################
## Plot together ##
###################

fs = 16

plot_data = pd.DataFrame()
for sector in top_sectors:
    country_dict = {'United Kingdom':'UK', 'South Korea':'S. Korea', 'Czech Republic':'Czechia', 'United States':'USA', 'Rest of the World':'RoW'}
    for item in change_sector[sector]['country'].unique():
        if item not in list(country_dict.keys()):
            country_dict[item] = item
            
    data_dict = {'oecd, figaro':'ICIO, Figaro', 'oecd, exio':'Exiobase, ICIO', 'oecd, gloria':'ICIO, Gloria', 
                 'figaro, exio':'Exiobase, Figaro', 'figaro, gloria':'Figaro, Gloria', 'exio, gloria':'Exiobase, Gloria'}
    
    change_sector[sector]['dataset'] = change_sector[sector]['dataset'].map(data_dict)
    change_sector[sector]['country'] = change_sector[sector]['country'].map(country_dict)
    
    temp = change_sector[sector][['country', 'dataset', 'RMSPE', 'percentage_sector', 'percentage_sector_c']]
    temp['sector'] = sector
    plot_data = plot_data.append(temp)
    

for sector in top_sectors:    
    
    temp = plot_data.loc[plot_data['sector'] == sector].sort_values(['percentage_sector', 'dataset'], ascending=False)
    
    fig, ax1 = plt.subplots(figsize=(17.5, 5))
    sns.stripplot(ax=ax1, data=temp, y='RMSPE', x='country', hue='dataset', size=7.5)
    plt.yscale('log')
    plt.title(sector, fontsize=fs)
    ax1.legend(bbox_to_anchor=(1.3, 1), fontsize=fs)
    ax1.tick_params(axis='y', labelsize=fs)
    
    ax1.tick_params(axis='x', labelsize=fs, rotation=90)
    ax1.set_ylabel('RMSPE (%)', fontsize=fs); 
    ax1.set_xlabel('', fontsize=fs); 
    
    ax2 = ax1.twinx()
    sns.lineplot(ax=ax2, data=temp, y='percentage_sector', x='country', color='k')
    ax2.tick_params(axis='y', labelsize=fs)
    ax2.set_ylabel('Sector CO2 of total CO2 (%)', fontsize=fs); 

    for i in range(len(temp['country'].unique())-1):
        ax1.axvline(i+0.5, color='k', linestyle=':')

    fig.tight_layout()
    plt.savefig(plot_filepath + 'stripplot_RMSPE_bysectors_' + sector +'.png', dpi=200, bbox_inches='tight')
    plt.show()