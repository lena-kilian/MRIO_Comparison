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
outputs_filepath = wd + 'ESCoE_Project/outputs/compare_all_outputs/'
plot_filepath = outputs_filepath + 'plots/'
    
# define number of top sectors to include
n = 10 

# plot params
fs = 16
pal = 'colorblind'
c_box = '#000000'
c_vlines = '#6d6d6d'
point_size = 30
scatter_size = 100
pal = 'tab10'
marker_list = ["o", "X", "s", "P"]


# Load Data
summary_industry = {}
summary_industry['Aggregation after Footprint Calculation'] = pickle.load(open(outputs_filepath + 'summary_industry.p', 'rb'))
summary_industry['Aggregation before Footprint Calculation'] = pickle.load(open(outputs_filepath + 'summary_industry_agg_all.p', 'rb'))

aggregations = list(summary_industry.keys())
datasets = summary_industry[aggregations[0]]['Total'].columns.tolist(); datasets.sort()
years = summary_industry[aggregations[0]]['Total'].index.levels[2].tolist()

data_comb = []
for i in range(len(datasets)):
    for j in range(i+1, len(datasets)):
        data_comb.append(datasets[i] + ', ' + datasets[j])



####################
## Industry Top n ##
####################

# get mean emissions by sector
for item in ['Total', 'Imports']:
    
    temp = summary_industry['Aggregation after Footprint Calculation'][item].sum(axis=0, level=['industry', 'year'])[datasets]
    order = pd.DataFrame(temp.mean(axis=0, level='industry').mean(1)).sort_values(0, ascending=False)
    order_list = ['Total'] + order.iloc[:n].index.tolist()
    
    fig, axs = plt.subplots(figsize=(10, 7.5), ncols=2, sharey=True, sharex=True)

    for i in range(2):
        agg = aggregations[i]
        sums = summary_industry[agg][item].sum(axis=0, level=['industry', 'year'])[datasets].unstack('year').T
        sums['Total'] = sums.sum(1)
        sums = sums.T.stack('year')
        sums = pd.DataFrame(sums.stack()).loc[order_list].reset_index().rename(columns={'level_2':'Data'})
        
        # plot    
        sns.pointplot(ax=axs[i], data=sums, x=0, y='industry', hue='Data', dodge=0.6, linestyles='', errorbar=None,
                      errwidth=0, markersize=point_size, palette=pal, markers=marker_list)
        
        axs[i].set_title(agg)
        axs[i].set_ylabel('')
        axs[i].set_xlabel('ktCO\N{SUBSCRIPT TWO}e')
        axs[i].set_xscale('log')
        for j in range(n+1):
            axs[i].axhline(0.5+j, c='k', linestyle=':')
        
    fig.tight_layout()
    plt.savefig(plot_filepath + 'pointplot_CO2_global_by_sector_GHG_agg&og_' + item + '.png', dpi=200, bbox_inches='tight')
    plt.show()


corr_agg = pd.DataFrame()
sums_agg = pd.DataFrame()
fig, axs = plt.subplots(figsize=(8, 4), ncols=2, sharey=True, sharex=True)
for i in range(2):
    item = ['Total', 'Imports'][i]
    
    plot_data = pd.DataFrame(summary_industry['Aggregation after Footprint Calculation'][item].stack())\
        .rename(columns={0:'Aggregation after Footprint Calculation'})
    temp = pd.DataFrame(summary_industry['Aggregation before Footprint Calculation'][item].stack())\
        .rename(columns={0:'Aggregation before Footprint Calculation'})
    
    plot_data = plot_data.join(temp).reset_index().rename(columns={'level_3':'Data'})
    plot_data = plot_data.sort_values('Data')
    
    # corr
    temp = plot_data.drop('year', axis=1).groupby('Data').corr().swaplevel(axis=0)\
        .loc['Aggregation before Footprint Calculation',  'Aggregation after Footprint Calculation'].T
    temp['Type'] = item
    temp['level'] = 'country & industry & year'
    corr_agg = corr_agg.append(temp)
    
    temp = plot_data.groupby(['industry', 'Data', 'year']).sum().groupby('Data').corr().swaplevel(axis=0)\
        .loc['Aggregation before Footprint Calculation',  'Aggregation after Footprint Calculation'].T
    temp['Type'] = item
    temp['level'] = 'industry & year'
    corr_agg = corr_agg.append(temp)
    
    temp = plot_data.groupby(['country', 'Data', 'year']).sum().groupby('Data').corr().swaplevel(axis=0)\
        .loc['Aggregation before Footprint Calculation',  'Aggregation after Footprint Calculation'].T
    temp['Type'] = item
    temp['level'] = 'country & year'
    corr_agg = corr_agg.append(temp)
    
    # sums
    temp = plot_data.set_index(['industry', 'country', 'year', 'Data']).mean(axis=0, level=['industry', 'country', 'Data'])\
        .sum(axis=0, level=['Data']).reset_index()
    temp['sum'] = 'all'
    temp['level'] = 'all'
    temp['Type'] = item
    sums_agg = sums_agg.append(temp)
    
    temp = plot_data.set_index(['industry', 'country', 'year', 'Data']).mean(axis=0, level=['industry', 'country', 'Data'])\
        .sum(axis=0, level=['industry', 'Data']).reset_index().rename(columns={'industry':'level'})
    temp['sum'] = 'industry'
    temp['Type'] = item
    sums_agg = sums_agg.append(temp)
    
    temp = plot_data.set_index(['industry', 'country', 'year', 'Data']).mean(axis=0, level=['industry', 'country', 'Data'])\
        .sum(axis=0, level=['country', 'Data']).reset_index().rename(columns={'country':'level'})
    temp['sum'] = 'country'
    temp['Type'] = item
    sums_agg = sums_agg.append(temp)
  
    #plot
    sns.scatterplot(ax=axs[i], data=plot_data, x='Aggregation after Footprint Calculation', y='Aggregation before Footprint Calculation', 
                    hue='Data', s=10, alpha=0.3)
    
    axs[i].set_title(item)
    axs[i].set_xscale('log')
    axs[i].set_yscale('log')
        
fig.tight_layout()
plt.savefig(plot_filepath + 'scatterplot_CO2_global_by_sector_GHG_agg&og_all.png', dpi=200, bbox_inches='tight')
plt.show()

sums_agg['diff'] = np.abs(sums_agg['Aggregation after Footprint Calculation'] - sums_agg['Aggregation before Footprint Calculation'])
sums_agg['diff pct'] =  sums_agg['diff']/sums_agg[['Aggregation after Footprint Calculation', 'Aggregation before Footprint Calculation']].mean(1) * 100
sums_agg = sums_agg.set_index(['Data', 'sum', 'level', 'Type']).unstack(['Data', 'Type'])
