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
import copy as cp
from sklearn.linear_model import LinearRegression

# set working directory
# make different path depending on operating system
if platform[:3] == 'win':
    wd = 'O://'
else:
    wd = r'/Volumes/a72/' 
    
    
fs = 16
pal = 'tab10'
c_box = '#000000'
c_vlines = '#B9B9B9'
point_size = 20

# define filepaths
data_filepath = wd + 'ESCoE_Project/data/'
emissions_filepath = wd + 'ESCoE_Project/data/Emissions/'
plot_filepath = wd + 'ESCoE_Project/outputs/compare_all_outputs/plots/'

co2_all = pickle.load(open(emissions_filepath + 'Emissions_aggregated_all.p', 'rb'))

error = 'rmse_pct' # 'rmspe'
corr_method = 'spearman' # 'pearson

datasets = list(co2_all.keys())
years = list(co2_all[datasets[0]].keys())

data_comb = ['ICIO, Figaro', 'Exiobase, ICIO', 'ICIO, Gloria', 'Exiobase, Figaro', 'Figaro, Gloria', 'Exiobase, Gloria']

country_dict = {'United Kingdom':'UK', 'Czech Republic':'Czechia', 'United States':'USA', 'Rest of the World':'RoW'}
data_dict = {'oecd':'ICIO', 'exio':'Exiobase', 'gloria':'Gloria', 'figaro':'Figaro'}

'''
# get openness of economy for ordering graphs
openness = pd.read_excel(data_filepath + 'lookups/lookup_trade_openness.xlsx', sheet_name='agg_data')
openness = openness.loc[openness['Countries'] != 'ROW Mean'].sort_values('Trade_openness_2018', ascending=False)

country_order = []
for c in openness['combined_name']:
    if c in list(country_dict.keys()):
        country_order.append(country_dict[c])
    else:
        country_order.append(c)

openness['country'] = country_order
'''



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


# get percentage imported
prop_im = pd.DataFrame((summary_im/summary * 100).mean(axis=0, level='country').mean(axis=1)).rename(columns={0:'Percentage CO2 imported'})

# Get means

mean_co2 = {'Total' : pd.DataFrame(summary.mean(axis=0, level='country').mean(axis=1)).rename(columns={0:'mean_co2'}), 
            'Imports' : pd.DataFrame(summary_im.mean(axis=0, level='country').mean(axis=1)).rename(columns={0:'mean_co2'})}


country_order = prop_im.sort_values('Percentage CO2 imported', ascending=False).index.tolist()

#############################
## Longitudinal footprints ##
#############################
'''
for country in summary.index.levels[0]:
    
    fig, axs = plt.subplots(figsize=(15, 5), ncols=2)
    # Total
    plot_data = summary.loc[country].stack().reset_index().rename(columns={'level_1':'Datasets', 0:'tCO2'})
    sns.lineplot(ax=axs[0], data=plot_data, x='year', y='tCO2', hue='Datasets')
    # Imports
    plot_data = summary_im.loc[country].stack().reset_index().rename(columns={'level_1':'Datasets', 0:'tCO2'})
    sns.lineplot(ax=axs[1], data=plot_data, x='year', y='tCO2', hue='Datasets')
    
    axs[0].set_title('Total')
    axs[1].set_title('Imports')
    
    fig.tight_layout()
    plt.savefig(plot_filepath + 'Lineplot_CO2_' + country + '_COLOUR.png', dpi=200, bbox_inches='tight')
    plt.show()
'''

fig, axs = plt.subplots(nrows=len(country_order), ncols=2, figsize=(10, 120))
for data in ['Total', 'Imports']:
    if data == 'Total':
        temp = cp.copy(summary)
        c = 0
    else:
        temp = cp.copy(summary_im)
        c=1
    
    for r in range(len(country_order)):
        country = country_order[r]
        plot_data = temp.loc[country].stack().reset_index().rename(columns={'level_1':'Datasets', 0:'tCO2'})
        sns.lineplot(ax=axs[r, c], data=plot_data, x='year', y='tCO2', hue='Datasets', legend=False)
        axs[r, c].set_title(country + ' - ' + data)
plt.savefig(plot_filepath + 'Lineplot_CO2_all.png', dpi=200, bbox_inches='tight')
plt.show()

'''
# corr with year

check_tot = summary.reset_index().groupby('country').corr().unstack()
check_tot.columns = [x[0] + ', ' + x[1] for x in check_tot.columns.tolist()]
check_tot = check_tot[['year, Exiobase', 'year, ICIO', 'year, Gloria', 'year, Figaro'] + data_comb]


check_imp = summary_im.reset_index().groupby('country').corr().unstack()
check_imp.columns = [x[0] + ', ' + x[1] for x in check_imp.columns.tolist()]
check_imp = check_imp[['year, Exiobase', 'year, ICIO', 'year, Gloria', 'year, Figaro'] + data_comb]
'''

# reg with year
reg_results = pd.DataFrame()
for country in summary.index.levels[0]:
    for ds in ['Exiobase', 'Gloria', 'ICIO', 'Figaro']:
    
        temp = summary.loc[country, ds].reset_index()
        
        regressor = LinearRegression()
        regressor.fit(temp[['year']], temp[ds])
        
        new = pd.DataFrame(index=[0])
        new['ds'] = ds
        new['country'] = country
        new['coef'] = regressor.coef_
        
        reg_results = reg_results.append(new)

reg_results = reg_results.set_index(['country', 'ds']).unstack()


reg_result2 = cp.copy(reg_results.droplevel(axis=1, level=0))
reg_result2['mean_co2'] = summary.reset_index('country').groupby('country').mean().mean(1)
reg_result2 = reg_result2.apply(lambda x: x/reg_result2['mean_co2'] *100)

temp = reg_result2.drop('mean_co2', axis=1).T.describe().T[['max', 'min']]
temp['crosses 0'] = False
temp.loc[(temp['max'] > 0) & (temp['min'] < 0), 'crosses 0'] = True

plot_data = reg_result2.drop('mean_co2', axis=1).join(temp[['crosses 0']]).set_index('crosses 0', append=True)\
    .stack().reset_index().rename(columns={0:'Average pct change'})
sns.scatterplot(data=plot_data, x='country', y='Average pct change', hue='crosses 0'); plt.axhline(0); plt.show()

#############################
## Longitudinal footprints ##
#############################

trend_data = cp.copy(summary).unstack('country').T
trend_data = trend_data.apply(lambda x: x/trend_data[2010]*100).T.stack().stack()\
    .reset_index().rename(columns={'level_2':'Data', 0:'Pct change (2010=100)'})
    
for country in country_order:
    plot_data = trend_data.loc[trend_data['country'] == country]
    sns.lineplot(data=plot_data, x='year', y='Pct change (2010=100)', hue='Data')
    plt.title(country)
    plt.axhline(100)
    plt.show()
    
    
corr_s = summary.reset_index('country').groupby('country').corr('spearman').unstack(level=1)
corr_s.columns = [x[0] + ', ' + x[1] for x in corr_s.columns]
corr_s = corr_s[data_comb]

plot_data = corr_s.stack().reset_index().rename(columns={'level_1':'Data', 0:'Corr'})
sns.boxplot(data=plot_data, x='Data', y='Corr'); plt.show()
sns.scatterplot(data=plot_data, x='country', y='Corr'); plt.show()


#######################
## Local correlation ##
#######################

maxmin = summary.reset_index('country').groupby('country').describe().swaplevel(axis=1)[['max', 'min']]

peaks = pd.DataFrame(index=country_order, columns=maxmin.swaplevel(axis=1).columns)
for country in country_order:
    for ds in list(data_dict.values()):
        temp = summary.loc[country, ds:ds]
        
        max_val = maxmin.loc[country, ('max', ds)]
        max_year = temp.loc[temp[ds] == max_val]
        peaks.loc[country, (ds, 'max')] = max_year.index.values[0]
        
        min_val = maxmin.loc[country, ('min', ds)]
        min_year = temp.loc[temp[ds] == min_val]
        peaks.loc[country, (ds, 'min')] = min_year.index.values[0]

peaks = peaks.astype(int)

plot_data = peaks.stack(level=[0, 1]).reset_index().rename(columns={'level_0':'country', 'level_1':'Data', 'level_2':'value', 0:'year'})
sns.boxplot(data=plot_data, x='country', y='year', hue='value'); plt.show()
sns.boxplot(data=plot_data, x='country', y='min'); plt.show()
