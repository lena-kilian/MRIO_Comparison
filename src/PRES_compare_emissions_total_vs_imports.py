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

order_var = 'gdp' # 'prop_order' 'openness'

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


####################
## Sort countries ##
####################

# get orders
# get percentage imported
prop_im = pd.DataFrame((summary_im / summary).mean(1).mean(axis=0, level=0)).rename(columns={0:'Percentage CO2 imported'})
prop_order = prop_im.sort_values('Percentage CO2 imported', ascending=False).index.tolist()

# get openness of economy for ordering graphs
openness = pd.read_excel(data_filepath + 'lookups/lookup_trade_openness.xlsx', sheet_name='agg_data')
openness = openness.loc[openness['Countries'] != 'ROW Mean'].sort_values('Trade_openness_2018', ascending=False)
openness = openness['combined_name'].tolist()

# sort by GDP
country_list = pd.DataFrame(co2_all[datasets[0]][years[0]].index.levels[0]).set_index(0)
country_list['index_test'] = 1
# GDP data https://data.worldbank.org/indicator/NY.GDP.MKTP.CD
gdp = pd.DataFrame(pd.read_csv(data_filepath + 'GDP/GDP.csv', header=2, index_col=0)[[str(x) for x in years]].mean(1)).dropna(0)\
    .rename(index={'Korea, Rep.':'South Korea', 'Slovak Republic':'Slovakia', 'Czechia':'Czech Republic', 'Russian Federation':'Russia', 'Turkiye':'Turkey'})
gdp = gdp.join(country_list, how='outer')
gdp['country'] = gdp.index.tolist()
gdp.loc[gdp['index_test'] != 1, 'country'] = 'Rest of the World'
gdp = gdp.groupby('country').sum()[0].sort_values(0, ascending=False).rename(index=country_dict).index.tolist()

country_order = eval(order_var)

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

percent_im['Dataset'] = percent_im['dataset'].map(data_dict)
percent_im = percent_im.sort_values('Dataset').set_index('country').rename(index=country_dict).loc[country_order].rename(index=country_dict).reset_index()

plot_data = mean_co2.stack().reset_index(); plot_data['Type'] = 'Total'
temp = mean_co2_im.stack().reset_index(); temp['Type'] = 'Imports'
plot_data = plot_data.append(temp).drop_duplicates()
plot_data.columns = ['country', 'dataset', 'CO2', 'Type']
plot_data.index = list(range(len(plot_data)))

plot_data['Dataset'] = plot_data['dataset'].map(data_dict)
plot_data = plot_data.sort_values('Dataset').set_index('country').rename(index=country_dict).loc[country_order].rename(index=country_dict).reset_index()


## Scatterplot
ms = 200

fig, axs = plt.subplots(nrows=2, figsize=(20, 10), sharex=True)

temp = cp.copy(plot_data)
temp.loc[temp['Type'] == 'Imports', 'CO2'] = 0
temp['Linetype'] = temp['Type'].map({'Total':'Total emissions', 'Imports':'Proportion imported (%)'})
temp = temp.loc[(temp['Type'] == 'Total') | (temp['country'] == 'India')]
temp['Country'] = '                     ' + temp['country']

sns.scatterplot(ax=axs[0], data=temp, x='country', y='CO2', style='Dataset', s=ms, c=['#000000']*len(temp))
axs[0].set_ylabel('Footprint (CO2)', fontsize=fs); 
axs[0].set_yscale('log')

sns.scatterplot(ax=axs[1], data=percent_im, y='pct_im', x='country', style='Dataset', s=ms, c=['#000000']*len(percent_im))
axs[1].tick_params(axis='y', labelsize=fs)
axs[1].set_ylabel('Emisions imported (%)', fontsize=fs); 

axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), fontsize=fs, ncol=len(temp['Dataset'].unique()), markerscale=3)
axs[1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), fontsize=fs, ncol=len(temp['Dataset'].unique()), markerscale=3)

for i in range(2):
    axs[i].tick_params(axis='y', labelsize=fs)
    axs[i].set_xlabel(' ')
    for c in range(len(temp['country'].unique())-1):
        axs[i].axvline(c+0.5, c=c_vlines, linestyle=':')
    
axs[1].set_xticklabels(temp['Country'].unique(), rotation=90, va='center', fontsize=fs); 
axs[1].xaxis.set_ticks_position('top') # the rest is the same

fig.tight_layout()
plt.savefig(plot_filepath + 'scatterplot_overview_bycountry_GHG.png', dpi=200, bbox_inches='tight')
plt.show()


## barplot
ms = 200

fig, axs = plt.subplots(nrows=2, figsize=(25, 10), sharex=True)

temp = cp.copy(plot_data)
temp.loc[temp['Type'] == 'Imports', 'CO2'] = 0
temp['Linetype'] = temp['Type'].map({'Total':'Total emissions', 'Imports':'Proportion imported (%)'})
temp = temp.loc[(temp['Type'] == 'Total')]
temp['Country'] = '                     ' + temp['country']

sns.barplot(ax=axs[0], data=temp, x='country', y='CO2', hue='Dataset')
axs[0].set_ylabel('Footprint (CO2)', fontsize=fs); 
axs[0].set_yscale('log')

sns.barplot(ax=axs[1], data=percent_im, y='pct_im', x='country', hue='Dataset')
axs[1].tick_params(axis='y', labelsize=fs)
axs[1].set_ylabel('Emisions imported (%)', fontsize=fs); 

axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), fontsize=fs, ncol=len(temp['Dataset'].unique()), markerscale=3)
axs[1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), fontsize=fs, ncol=len(temp['Dataset'].unique()), markerscale=3)

for i in range(2):
    axs[i].tick_params(axis='y', labelsize=fs)
    axs[i].set_xlabel(' ')
    for c in range(len(temp['country'].unique())-1):
        axs[i].axvline(c+0.5, c=c_vlines, linestyle=':')
    
axs[1].set_xticklabels(temp['Country'].unique(), rotation=90, va='center', fontsize=fs); 
axs[1].xaxis.set_ticks_position('top') # the rest is the same

fig.tight_layout()
plt.savefig(plot_filepath + 'barplot_overview_bycountry_GHG.png', dpi=200, bbox_inches='tight')
plt.show()




# pointplot

ms = 200

fig, axs = plt.subplots(nrows=2, figsize=(25, 10), sharex=True)

temp = cp.copy(plot_data)
temp.loc[temp['Type'] == 'Imports', 'CO2'] = 0
temp['Linetype'] = temp['Type'].map({'Total':'Total emissions', 'Imports':'Proportion imported (%)'})
temp = temp.loc[(temp['Type'] == 'Total')]
temp['Country'] = '                     ' + temp['country']

temp2 = temp.groupby(['country', 'Country']).describe().stack(level=1).loc[temp['country'].unique()].reset_index()
temp2 = temp2.loc[temp2['level_2'].isin(['min', 'max']) == True]


sns.pointplot(ax=axs[0], data=temp, x='Country', y='CO2', color='#000000', linestyles="", errorbar='sd')
sns.scatterplot(ax=axs[0], data=temp2, x='Country', y='CO2', color='#000000', s=150, marker='_')
axs[0].set_ylabel('Footprint (CO2)', fontsize=fs); 
axs[0].set_yscale('log')

temp = cp.copy(percent_im)
temp['Country'] = '                     ' + temp['country']

temp2 = temp.groupby(['country', 'Country']).describe().stack(level=1).loc[temp['country'].unique()].reset_index()
temp2 = temp2.loc[temp2['level_2'].isin(['min', 'max']) == True]

sns.pointplot(ax=axs[1], data=temp, x='Country', y='pct_im', color='#000000', linestyles="", errorbar='sd')
sns.scatterplot(ax=axs[1], data=temp2, x='Country', y='pct_im', color='#000000', s=150, marker='_')
axs[1].tick_params(axis='y', labelsize=fs)
axs[1].set_ylabel('Emisions imported (%)', fontsize=fs); 

for i in range(2):
    axs[i].tick_params(axis='y', labelsize=fs)
    axs[i].set_xlabel(' ')
    for c in range(len(temp['country'].unique())-1):
        axs[i].axvline(c+0.5, c=c_vlines, linestyle=':')
    
axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=90, va='center', fontsize=fs); 
axs[1].xaxis.set_ticks_position('top') # the rest is the same

fig.tight_layout()
plt.savefig(plot_filepath + 'pointplot_overview_bycountry_GHG.png', dpi=200, bbox_inches='tight')
plt.show()

