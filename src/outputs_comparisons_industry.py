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
    
fs = 16
pal = 'tab10'
c_box = '#000000'
c_vlines = '#B9B9B9'
point_size = 20

order_var = 'gdp' # 'prop_order'
country_order = pickle.load(open(outputs_filepath + 'country_order.p', 'rb'))[order_var]


co2_all = pickle.load(open(emissions_filepath + 'Emissions_aggregated_all.p', 'rb'))

error = 'rmse_pct' # 'rmspe'


datasets = list(co2_all.keys())
years = list(co2_all[datasets[0]].keys())

# plot params
fs = 16
pal = 'tab10'
c_box = '#000000'
c_vlines = '#6d6d6d'
point_size = 20
scatter_size = 100
pal = 'tab10'



fig, axs = plt.subplots(nrows=len(data_comb), figsize=(10, 10), sharex=True)#, sharey=True)
for i in range(len(data_comb)):
    item = data_comb[i]
    plot_data = corr_all.loc[corr_all['Data'] == item]
    sns.histplot(ax=axs[i], data=plot_data, x="Spearman's Rho", hue='Type', binwidth=0.025)
    #sns.kdeplot(ax=axs[i], data=plot_data, x="Spearman's Rho", hue='Type')
    axs[i].set_title(item)
    axs[i].set_xlim(0, 1)
    #axs[i].set_xlim(-1, 1)
fig.tight_layout()
plt.savefig(plot_filepath + 'histplot_CO2_sector_corr_by_data_GHG.png', dpi=200, bbox_inches='tight')
plt.show() 

# Scatterplot
plot_data = corr.set_index('Data').loc[data_comb].reset_index().rename(columns={'Total':"Spearman's Rho"})
plot_data_im = corr_im.set_index('Data').loc[data_comb].reset_index().rename(columns={'Imports':"Spearman's Rho"})
plot_data['Country'] = '                     ' + plot_data['country']
plot_data_im['Country'] = '                     ' + plot_data_im['country']

fig, axs = plt.subplots(nrows=2, figsize=(20, 10), sharex=True)
# total
sns.scatterplot(ax=axs[0], data=plot_data, x='country', y="Spearman's Rho", hue='Data', s=120)
# imports
sns.scatterplot(ax=axs[1], data=plot_data_im, x='country', y="Spearman's Rho", hue='Data',  s=120, legend=False)

axs[1].xaxis.set_ticks_position('top') # the rest is the same
axs[1].set_xticklabels(plot_data['Country'].unique(), rotation=90, va='center', fontsize=fs); 
axs[0].legend(bbox_to_anchor=(1,1))

for i in [0.3, 0.5, 0.7]:
    axs[0].axhline(i, c='k', linestyle=':')
    axs[1].axhline(i, c='k', linestyle=':')
    
fig.tight_layout()
plt.savefig(plot_filepath + 'scatterplot_CO2_sector_corr_by_country_GHG.png', dpi=200, bbox_inches='tight')
plt.show()

####################
## Industry Top 5 ##
####################

# get mean emissions by sector and country
n = 11

# total
sums = summary.sum(axis=0, level=['industry', 'year']).unstack('year').T
sums['Total'] = sums.sum(1)
sums = sums.T.stack('year')

order = pd.DataFrame(sums.mean(axis=0, level='industry').mean(1)).sort_values(0, ascending=False)
order['cumsum'] = order[0].cumsum()
order['cumpct'] = order['cumsum'] / order[0].sum() * 100
order_list = order.iloc[:n].index.tolist()

sums = pd.DataFrame(sums.stack())

#imports
sums_im = summary_im.sum(axis=0, level=['industry', 'year']).unstack('year').T
sums_im['Total'] = sums_im.sum(1)
sums_im = sums_im.T.stack('year')
order_im = pd.DataFrame(sums_im.mean(axis=0, level='industry').mean(1)).sort_values(0, ascending=False)
order_im['cumsum'] = order_im[0].cumsum()
order_im['cumpct'] = order_im['cumsum'] / order_im[0].sum() * 100
order_im_list = order_im.iloc[:n].index.tolist()

sums_im = pd.DataFrame(sums_im.stack())


#plot
fig, axs = plt.subplots(figsize=(17, 10), ncols=2)#, sharey=True)
sns.barplot(ax=axs[0], data = sums.loc[order_list].reset_index(), x=0, y='industry', hue='level_2')
sns.barplot(ax=axs[1], data = sums_im.loc[order_im_list].reset_index(), x=0, y='industry', hue='level_2')
for j in range(2):
    axs[j].set_title(['Total', 'Imports'][j])
    axs[j].set_ylabel('')
    axs[j].set_xlabel('tCO2e')
    axs[j].set_xscale('log')
    for i in range(n):
        axs[j].axhline(0.5+i, c='k', linestyle=':')
        
fig.tight_layout()
plt.savefig(plot_filepath + 'barplot_CO2_global_by_sector_GHG.png', dpi=200, bbox_inches='tight')
plt.show()

        
######################
## Pct distribution ##
######################
n = 10

# total
pct = summary.sum(axis=0, level=['industry', 'year']).unstack('year').apply(lambda x: x/x.sum() * 100).stack('year')

order = pd.DataFrame(pct.mean(axis=0, level='industry').mean(1)).sort_values(0, ascending=False)
order['cumsum'] = order[0].cumsum()
order['cumpct'] = order['cumsum'] / order[0].sum() * 100
order_list = order.iloc[:n].index.tolist()

pct = pd.DataFrame(pct.stack())

#imports
pct_im = summary_im.sum(axis=0, level=['industry', 'year']).unstack('year').apply(lambda x: x/x.sum() * 100).stack('year')
order_im = pd.DataFrame(pct_im.mean(axis=0, level='industry').mean(1)).sort_values(0, ascending=False)
order_im['cumsum'] = order_im[0].cumsum()
order_im['cumpct'] = order_im['cumsum'] / order_im[0].sum() * 100
order_im_list = order_im.iloc[:n].index.tolist()

pct_im = pd.DataFrame(pct_im.stack())


#plot
fig, axs = plt.subplots(figsize=(17, 10), ncols=2)#, sharey=True)
sns.barplot(ax=axs[0], data = pct.loc[order_list].reset_index(), x=0, y='industry', hue='level_2')
sns.barplot(ax=axs[1], data = pct_im.loc[order_im_list].reset_index(), x=0, y='industry', hue='level_2')
for j in range(2):
    axs[j].set_title(['Total', 'Imports'][j])
    axs[j].set_ylabel('')
    axs[j].set_xlabel('Percentage of tCO2e')
    for i in range(n):
        axs[j].axhline(0.5+i, c='k', linestyle=':')

fig.tight_layout()
plt.savefig(plot_filepath + 'barplot_CO2_global_by_sector_GHG_pct.png', dpi=200, bbox_inches='tight')
plt.show()