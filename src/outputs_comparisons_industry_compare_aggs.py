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
summary_industry = {'original':pickle.load(open(outputs_filepath + 'summary_industry.p', 'rb')),
                    'agg_construction':pickle.load(open(outputs_filepath + 'summary_industry_agg_construction.p', 'rb')),
                    'agg_all':pickle.load(open(outputs_filepath + 'summary_industry_agg_all.p', 'rb'))
                    }

corr = {'original':pickle.load(open(outputs_filepath + 'corr_industry.p', 'rb')),
        'agg_construction':pickle.load(open(outputs_filepath + 'corr_industry_agg_construction.p', 'rb')),
        'agg_all':pickle.load(open(outputs_filepath + 'corr_industry_agg_all.p', 'rb'))
        }

datasets = summary_industry['original']['Total'].columns.tolist(); datasets.sort()
years = summary_industry['original']['Total'].index.levels[2].tolist()

data_comb = []
for i in range(len(datasets)):
    for j in range(i+1, len(datasets)):
        data_comb.append(datasets[i] + ', ' + datasets[j])

#################
## Combine all ##
#################

# Summary
summary_all = pd.DataFrame()
for version in list(summary_industry.keys()):
    for item in list(summary_industry[version].keys()):
        temp = cp.copy(summary_industry[version][item]).reset_index()
        temp['version'] = version
        temp['type'] = item
        summary_all = summary_all.append(temp)
        
    
# Correlation
corr_all = pd.DataFrame()
for version in list(corr.keys()):
    for item in list(corr[version].keys()):
        temp = cp.copy(corr[version][item]).reset_index()
        temp['version'] = version
        temp['type'] = item
        corr_all = corr_all.append(temp)
corr_all = corr_all.drop('index', axis=1)
corr_all.index = list(range(len(corr_all)))
        
#################
## Correlation ##
#################

# Plot Histogram
fig, axs = plt.subplots(nrows=len(data_comb), ncols=2, figsize=(10, 10), sharex=True)#, sharey=True)
for c in range(2):
    item = ['Total', 'Imports'][c]
    for r in range(len(data_comb)):
        plot_data = corr_all.loc[(corr_all['type'] == item) & (corr_all['Data'] == data_comb[r])]
        sns.histplot(ax=axs[r, c], data=plot_data, x='spearman', binwidth=0.025, hue='version')
        axs[r, c].set_title(data_comb[r])
        axs[r, c].set_xlim(0, 1)
    axs[r, c].set_xlabel("Spearman's Rho")
fig.tight_layout()
#plt.savefig(plot_filepath + 'histplot_CO2_sector_corr_by_data_GHG_agg_all.png', dpi=200, bbox_inches='tight')
plt.show() 


# Plot boxplot
fig, axs = plt.subplots(figsize=(10, 5), nrows=2)
for r in range(2):
    item = ['Total', 'Imports'][c]
    plot_data = corr_all.loc[(corr_all['type'] == item)]
    sns.boxplot(ax=axs[r], data=plot_data, y='spearman', x='Data', hue='version')
    axs[r].legend(bbox_to_anchor=(1,1))
    axs[r].set_xlabel('')
    axs[r].set_ylabel("Spearman's Rho")
fig.tight_layout()
#plt.savefig(plot_filepath + 'histplot_CO2_sector_corr_by_data_GHG_agg_all.png', dpi=200, bbox_inches='tight')
plt.show() 


####################
## Industry Top n ##
####################

order_full = summary_all.groupby(['industry' ,'country', 'type']).mean()[datasets].mean(1).sum(axis=0, level=[0, 2]).reset_index()
l = len(summary_industry)

# get mean emissions by sector
for r in range(2):
    item = ['Total', 'Imports'][r]
    
    order = order_full.loc[order_full['type'] == item].sort_values(0, ascending=False)['industry'][:n]
    
    fig, axs = plt.subplots(figsize=(l*5, 8), ncols=l, sharex=True, sharey=True)
    for c in range(l):
        version = list(summary_industry.keys())[c]
    
        plot_data = summary_industry[version][item].sum(axis=0, level=[0, 2]).mean(axis=0, level=0).loc[order].stack().reset_index()
        
        # plot    
        #sns.stripplot(ax=axs[i], data = sums, x=0, y='industry', hue='level_2', dodge=True, alpha=.2, , palette=pal)
        #sns.pointplot(ax=axs[c], data=plot_data, x=0, y='industry', hue='level_1', dodge=0.6, linestyles='', errorbar=None,
        #              markersize=point_size, palette=pal, markers=marker_list)
        
        sns.barplot(ax=axs[c], data=plot_data, x=0, y='industry', hue='level_1', palette=pal)
        
        axs[c].set_title(version + ' - ' + item)
        axs[c].set_ylabel('')
        axs[c].set_xlabel('tCO2e')
        axs[c].set_xscale('log')
        for j in range(n):
            axs[c].axhline(0.5+j, c='k', linestyle=':')
            
    fig.tight_layout()
    plt.savefig(plot_filepath + 'barplot_CO2_global_by_sector_GHG_agg_compare' + item + '.png', dpi=200, bbox_inches='tight')
    plt.show()
    
########################
## Compare by version ##
########################

order_full = summary_all.groupby(['industry' ,'country', 'type']).mean()[datasets].mean(1).sum(axis=0, level=[0, 2]).reset_index()
l = len(summary_industry)

sums = summary_all.groupby(['industry' ,'country', 'type', 'version']).mean()[datasets].sum(axis=0, level=[0, 2, 3]).stack().reset_index()
# get mean emissions by sector
fig, axs = plt.subplots(figsize=(16, int(n*1.5)), ncols=2, nrows=n, sharex=True)
for c in range(2):
    item = ['Total', 'Imports'][c]
    
    order = order_full.loc[order_full['type'] == item].sort_values(0, ascending=False)['industry'].tolist()
    
    for r in range(n):
        industry = order[r]
    
        plot_data = sums.loc[(sums['industry'] == industry) & (sums['type'] == item)]
        
        sns.barplot(ax=axs[r, c], data=plot_data, x=0, y='level_3', hue='version', palette=pal)

        #axs[r, c].set_title(version + ' - ' + item)
        axs[r, c].set_ylabel(industry)
        #axs[r, c].set_xscale('log')
        
        #axs[r, c].get_legend().remove()
        axs[r, c].set_xlabel('')
        
    axs[r, c].set_xlabel('(tCO2e)')
    axs[0, c].set_title(item)
            
fig.tight_layout()
plt.savefig(plot_filepath + 'barplot_CO2_global_by_sector_compare_agg_versions.png', dpi=200, bbox_inches='tight')
plt.show()

