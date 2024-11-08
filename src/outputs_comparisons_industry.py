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
from matplotlib.cm import get_cmap
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

# define when aggregation happends
agg_vars = ['_agg_after']#, '_agg_before']

# plot params
fs = 16
pal = 'colorblind'
c_box = '#000000'
c_vlines = '#6d6d6d'
point_size = 30
scatter_size = 100
pal = 'tab10'
marker_list = ["o", "X", "s", "P"]

for agg_var in agg_vars:
    # Load Data
    summary_industry = pickle.load(open(outputs_filepath + 'summary_industry' + agg_var + '.p', 'rb'))
    corr = pickle.load(open(outputs_filepath + 'corr_industry' + agg_var + '.p', 'rb'))
    
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
            axs[r, c].axvline(plot_data.median().values, c='k', linestyle=':', linewidth=2)
            axs[r, c].set_title(data_comb[r])
            axs[r, c].set_xlim(0, 1)
            
            print(item, data_comb[r], plot_data.median().values)
        axs[r, c].set_xlabel("Spearman's Rho", fontsize=fs)
    fig.tight_layout()
    plt.savefig(plot_filepath + 'histplot_CO2_sector_corr_by_data_GHG' + agg_var + '.png', dpi=200, bbox_inches='tight')
    plt.show() 
    
    
    # Plot Scatterplot
    fig, axs = plt.subplots(nrows=len(data_comb), ncols=2, figsize=(8, 10), sharex=True, sharey=True)
    for c in range(2):
        item = ['Total', 'Imports'][c]
        for r in range(len(data_comb)):
            plot_data = corr[item].loc[corr[item]['Data'] == data_comb[r]]
            sns.histplot(ax=axs[r, c], data=plot_data, x='spearman', binwidth=0.05, color=get_cmap(pal)(c), alpha=0.5)
            axs[r, c].axvline(plot_data.median().values, c='k', linestyle=':', linewidth=2)
            axs[r, c].set_ylabel(data_comb[r].replace(', ', ',\n'), fontsize=fs)
            axs[r, c].set_xlim(0, 1)
            #y_labels =[int(y) for y in axs[r, c].get_yticks()]
            y_labels = [0, 10, 20, 30, 40]
            #print(y_labels)
            axs[r, c].set_yticklabels(y_labels, fontsize=fs); 
        axs[r, c].set_xlabel("Spearman's Rho", fontsize=fs)
        x_labels = [round(x, 2) for x in axs[r, c].get_xticks()]
        axs[r, c].set_xticklabels(x_labels, fontsize=fs); 
        axs[0, c].set_title(item, fontsize=fs)
    fig.tight_layout()
    plt.savefig(plot_filepath + 'histplot_CO2_sector_corr_by_data_GHG' + agg_var + '.png', dpi=200, bbox_inches='tight')
    plt.show() 
    
    ##################
    ## Industry All ##
    ##################
    
    # get mean emissions by sector
    
    
    
    sums = summary_industry['Total'].sum(axis=0, level=['industry', 'year'])[datasets].unstack('year').T
    sums['Total'] = sums.sum(1)
    sums = sums.T.stack('year')    
    order = pd.DataFrame(sums.mean(axis=0, level='industry').mean(1)).sort_values(0, ascending=False)
    order_list = order.index.tolist()
    
    order_list.sort()
    check = pd.DataFrame(order_list)
    
    fig, axs = plt.subplots(figsize=(10, 16), ncols=2, sharey=True)
    for i in range(2):
        item = ['Total', 'Imports'][i]
        
        sums = summary_industry[item].sum(axis=0, level=['industry', 'year'])[datasets].unstack('year').T
        sums['Total'] = sums.sum(1)
        sums = sums.T.stack('year')
    
        sums = pd.DataFrame(sums.stack()).loc[order_list].reset_index().rename(columns={'level_2':'Data'})
        
        # plot    
        #sns.stripplot(ax=axs[i], data = sums, x=0, y='industry', hue='level_2', dodge=True, alpha=.2, , palette=pal)
        sns.pointplot(ax=axs[i], data = sums, x=0, y='industry', hue='Data', dodge=0.6, linestyles='', errorbar=None,
                      errwidth=0, markersize=point_size, palette=pal, markers=marker_list)
        
        axs[i].set_title(item)
        axs[i].set_ylabel('')
        axs[i].set_xlabel('ktCO\N{SUBSCRIPT TWO}e')
        axs[i].set_xscale('log')
        for j in range(len(order_list)):
            axs[i].axhline(0.5+j, c='k', linestyle=':')
        
    fig.tight_layout()
    plt.savefig(plot_filepath + 'pointplot_CO2_global_by_sector_GHG_ALL' + agg_var + '.png', dpi=200, bbox_inches='tight')
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
    
        sums = pd.DataFrame(sums.stack()).loc[order_list].reset_index().rename(columns={'level_2':'Data'})
        
        # plot    
        #sns.stripplot(ax=axs[i], data = sums, x=0, y='industry', hue='level_2', dodge=True, alpha=.2, , palette=pal)
        sns.pointplot(ax=axs[i], data = sums, x=0, y='industry', hue='Data', dodge=0.6, linestyles='', errorbar=None,
                      errwidth=0, markersize=point_size, palette=pal, markers=marker_list)
        
        axs[i].set_title(item)
        axs[i].set_ylabel('')
        axs[i].set_xlabel('ktCO\N{SUBSCRIPT TWO}e')
        axs[i].set_xscale('log')
        for j in range(n+1):
            axs[i].axhline(0.5+j, c='k', linestyle=':')
        
    fig.tight_layout()
    plt.savefig(plot_filepath + 'pointplot_CO2_global_by_sector_GHG' + agg_var + '.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    
    # industry means
    industry_order = pd.DataFrame()
    industry_prop = pd.DataFrame()
    industry_global = pd.DataFrame()
    for item in ['Total', 'Imports']:
        temp = summary_industry[item].mean(axis=0, level=[0, 1]).unstack('country')
        
        # order
        temp2 = cp.copy(temp)
        for col in temp2.columns.tolist():
            temp2 = temp2.sort_values(col, ascending=False)
            temp2[col] = list(range(len(temp2)))
        temp2['Data'] = item
        industry_order = industry_order.append(temp2.set_index('Data', append=True).stack(level=1).reset_index())
        
        # prop
        temp2 = temp.apply(lambda x: x/x.sum() * 100).stack(level=1).fillna(0)
        temp2['Data'] = item
        industry_prop = industry_prop.append(temp2.reset_index())
        
        # global
        temp2 = temp.sum(axis=1, level=0)
        temp2.columns = pd.MultiIndex.from_arrays([['GHG'] * len(temp2.columns), temp2.columns.tolist()])
        for col in temp2.columns.levels[1].tolist():
            temp2 = temp2.sort_values(('GHG', col), ascending=False)
            temp2[('order', col)] = list(range(len(temp2)))
        temp2['Data'] = item
        industry_global = industry_global.append(temp2.reset_index())
        
        
    industry_global_summary = industry_global.set_index(['industry', 'Data'])['GHG'].unstack().fillna(0).mean(axis=1, level=1)
    industry_global_summary2 = industry_global.set_index(['industry', 'Data'])['GHG'].unstack().fillna(0).swaplevel(axis=1)
    industry_global_summary2[('Total', 'aaa_mean')] = industry_global_summary2['Total'].mean(1)
    industry_global_summary2[('Imports', 'aaa_mean')] = industry_global_summary2['Imports'].mean(1)
    
    industry_global_summary2 = industry_global_summary2.reindex(sorted(industry_global_summary2.columns), axis=1)
    industry_global_summary_prop = industry_global_summary2.apply(lambda x: x/x.sum()*100)
    
    industry_order_top = pd.DataFrame(industry_order.set_index(['industry', 'country', 'Data']).stack())
    industry_order_top = industry_order_top.loc[industry_order_top[0] < n]
    industry_order_top[0] = 1
    industry_order_top = industry_order_top.unstack(level=3).fillna(0).droplevel(axis=1, level=0)
    industry_order_top['Count'] = industry_order_top.sum(1)
    
    industry_order_top_total = cp.copy(industry_order_top)
    for d in datasets:
        industry_order_top_total[d] = industry_order_top_total[d].map({0:'', 1:d + ', '})
    industry_order_top_total['Datasets'] = industry_order_top_total['Count'].astype(str) + ': ' + industry_order_top_total[datasets].sum(1).str[:-2]
    industry_order_top_total = industry_order_top_total[['Datasets']].unstack(level=[2,1]).droplevel(axis=1, level=0)
    
    industry_order_top_imports = industry_order_top_total['Imports']
    industry_order_top_total = industry_order_top_total['Total']
 
    industry_order_top_summary = pd.DataFrame(industry_order.set_index(['industry', 'country', 'Data']).stack())
    industry_order_top_summary = industry_order_top_summary.loc[industry_order_top_summary[0] < n]
    industry_order_top_summary[0] = 1
    industry_order_top_summary = industry_order_top_summary.sum(axis=0, level=[0, 2, 3]).unstack(level=[1, 2]).droplevel(axis=1, level=0).fillna(0)
    industry_order_top_summary[('Total', 'summary')] = industry_order_top_summary['Total'].mean(1)
    industry_order_top_summary[('Imports', 'summary')] = industry_order_top_summary['Imports'].mean(1)
    
    industry_prop_top = pd.DataFrame(industry_prop.set_index(['industry', 'country', 'Data']).stack())
    industry_prop_top = industry_prop_top.loc[industry_prop_top[0] >= 10].unstack(level=[2, 3]).droplevel(axis=1, level=0).fillna(0)
    industry_prop_top[('summary', 'Total_mean')] = industry_prop_top['Total'].mean(1)
    industry_prop_top[('summary', 'Imports_mean')] = industry_prop_top['Imports'].mean(1)   
    industry_prop_top[('summary', 'Total_sd')] = industry_prop_top['Total'].std(1)
    industry_prop_top[('summary', 'Imports_sd')] = industry_prop_top['Imports'].std(1)   
    #industry_prop_top = industry_prop_top['summary']
    
    

