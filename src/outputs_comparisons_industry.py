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
outputs_filepath = wd + 'ESCoE_Project/outputs/compare_all_outputs/'
plot_filepath = outputs_filepath + 'plots/'
    
# define number of top sectors to include
n = 10 

# plot params
fs = 16
pal = 'tab10'
c_box = '#000000'
c_vlines = '#6d6d6d'
point_size = 20
scatter_size = 100
pal = 'tab10'




# Load Data
summary_industry = pickle.load(open(outputs_filepath + 'summary_industry.p', 'rb'))
corr = pickle.load(open(outputs_filepath + 'corr_industry.p', 'rb'))

datasets = summary_industry['Total'].columns.tolist(); datasets.sort()
years = summary_industry['Total'].index.levels[2].tolist()

data_comb = []
for i in range(len(datasets)):
    for j in range(i+1, len(datasets)):
        data_comb.append(datasets[i] + ', ' + datasets[j])

#################
## Correlation ##
#################

# Plot Histogram
fig, axs = plt.subplots(nrows=len(data_comb), ncols=2, figsize=(10, 10), sharex=True)#, sharey=True)
for c in range(2):
    item = ['Total', 'Imports'][c]
    for r in range(len(data_comb)):
        plot_data = corr[item].loc[corr[item]['Data'] == data_comb[r]]
        sns.histplot(ax=axs[r, c], data=plot_data, x='spearman', binwidth=0.025)
        axs[r, c].set_title(data_comb[r])
        axs[r, c].set_xlim(0, 1)
    axs[r, c].set_xlabel("Spearman's Rho")
fig.tight_layout()
plt.savefig(plot_filepath + 'histplot_CO2_sector_corr_by_data_GHG.png', dpi=200, bbox_inches='tight')
plt.show() 


# Plot boxplot
plot_data = corr['Total']; plot_data['Type'] = 'Total'
temp = corr['Imports']; temp['Type'] = 'Imports'
plot_data = plot_data.append(temp)

fig, ax = plt.subplots(figsize=(20, 5))
sns.boxplot(ax=ax, data=plot_data, x='Data', y='spearman', hue='Type', showfliers=False, palette=pal)
ax.set_xlabel('')
ax.set_ylabel("Spearman's Rho", fontsize=fs)
ax.tick_params(axis='y', labelsize=fs)
  
ax.set_xticklabels(ax.get_xticklabels(), fontsize=fs); 

for c in range(len(plot_data['Data'].unique())):
    ax.axvline(c+0.5, c=c_vlines, linestyle=':')
    
ax.axhline(0, c=c_vlines)

fig.tight_layout()
plt.savefig(plot_filepath + 'coxplot_correlation_bydata_industry_GHG.png', dpi=200, bbox_inches='tight')
plt.show()


# Plot Scatterplot
fig, axs = plt.subplots(nrows=len(data_comb), ncols=2, figsize=(10, 10), sharex=True)#, sharey=True)
for c in range(2):
    item = ['Total', 'Imports'][c]
    for r in range(len(data_comb)):
        plot_data = corr[item].loc[corr[item]['Data'] == data_comb[r]]
        sns.histplot(ax=axs[r, c], data=plot_data, x='spearman', binwidth=0.025)
        axs[r, c].set_title(data_comb[r])
        axs[r, c].set_xlim(0, 1)
    axs[r, c].set_xlabel("Spearman's Rho")
fig.tight_layout()
plt.savefig(plot_filepath + 'scatterplot_CO2_sector_corr_by_data_GHG.png', dpi=200, bbox_inches='tight')
plt.show() 

####################
## Industry Top n ##
####################

# get mean emissions by sector
cumsum_industry = {}
fig, axs = plt.subplots(figsize=(10, 8), ncols=2)#, sharey=True)
for i in range(2):
    item = ['Total', 'Imports'][i]
    
    sums = summary_industry[item].sum(axis=0, level=['industry', 'year'])[datasets].unstack('year').T
    sums['Total'] = sums.sum(1)
    sums = sums.T.stack('year')
    
    order = pd.DataFrame(sums.mean(axis=0, level='industry').mean(1)).sort_values(0, ascending=False)
    order['cumsum'] = order[0].cumsum()
    order['cumpct'] = order['cumsum'] / order[0].sum() * 100
    
    cumsum_industry[item] = order
    
    order_list = order.iloc[:n+1].index.tolist()

    sums = pd.DataFrame(sums.stack()).loc[order_list].reset_index()
    
    # plot    
    #sns.stripplot(ax=axs[i], data = sums, x=0, y='industry', hue='level_2', dodge=True, alpha=.2)
    sns.pointplot(ax=axs[i], data = sums, x=0, y='industry', hue='level_2', dodge=0.6, linestyles='', errorbar='sd',
                  markersize=point_size)
    
    axs[i].set_title(item)
    axs[i].set_ylabel('')
    axs[i].set_xlabel('tCO2e')
    axs[i].set_xscale('log')
    for j in range(n+1):
        axs[i].axhline(0.5+j, c='k', linestyle=':')
    
fig.tight_layout()
plt.savefig(plot_filepath + 'pointplot_CO2_global_by_sector_GHG.png', dpi=200, bbox_inches='tight')
plt.show()


fig, axs = plt.subplots(figsize=(10, 8), ncols=2)#, sharey=True)
for i in range(2):
    item = ['Total', 'Imports'][i]
    
    sums = summary_industry[item].sum(axis=0, level=['industry', 'year'])[datasets].unstack('year').T
    sums['Total'] = sums.sum(1)
    sums = sums.T.stack('year')
    
    order = pd.DataFrame(sums.mean(axis=0, level='industry').mean(1)).sort_values(0, ascending=False)
    order['cumsum'] = order[0].cumsum()
    order['cumpct'] = order['cumsum'] / order[0].sum() * 100
    order_list = order.iloc[:n+1].index.tolist()

    sums = pd.DataFrame(sums.stack()).loc[order_list].reset_index()
    
    # plot    
    sns.barplot(ax=axs[i], data = sums, x=0, y='industry', hue='level_2')
    
    axs[i].set_title(item)
    axs[i].set_ylabel('')
    axs[i].set_xlabel('tCO2e')
    axs[i].set_xscale('log')
    for j in range(n+1):
        axs[i].axhline(0.5+j, c='k', linestyle=':')
    
fig.tight_layout()
plt.savefig(plot_filepath + 'barplot_CO2_global_by_sector_GHG.png', dpi=200, bbox_inches='tight')
plt.show()