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
from scipy.interpolate import make_interp_spline

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
    sns.lmplot(data=plot_data, x='year', y='tCO2', hue='Datasets')
    # Imports
    plot_data = summary_im.loc[country].stack().reset_index().rename(columns={'level_1':'Datasets', 0:'tCO2'})
    sns.lineplot(ax=axs[1], data=plot_data, x='year', y='tCO2', hue='Datasets')
    
    axs[0].set_title('Total')
    axs[1].set_title('Imports')
    
    fig.tight_layout()
    plt.savefig(plot_filepath + 'Lineplot_CO2_' + country + '_GHG.png', dpi=200, bbox_inches='tight')
    plt.show()
'''

fig, axs = plt.subplots(figsize=(15, 120), ncols=2, nrows=len(country_order))
for r in range(len(country_order)):
    country = country_order[r]
    for c in range(2):
        val = ['Total', 'Imports'][c]
        
        if val == 'Total':
            plot_data = summary.loc[country].reset_index()
        else:
            plot_data = summary_im.loc[country].reset_index()
            
        plot_data_new = pd.DataFrame()
        for ds in ['ICIO', 'Exiobase', 'Gloria', 'Figaro']:
        
            x = plot_data[ds].tolist()
            mean_x = [(x[0]+x[1])/2]
            for j in range(1, len(x)-1):
                mean_x.append((x[j-1] + x[j] + x[j+1])/3)
            mean_x.append((x[-2]+x[-1])/2)

            mean_x = np.array(mean_x)
            y = np.array(plot_data['year'])
             
            X_Y_Spline = make_interp_spline(y, mean_x)
        
            temp = pd.DataFrame(columns=['year', 'GHG (tCO2e)'])
            temp['year'] = np.linspace(y.min(), y.max(), 80)
            temp['GHG (tCO2e)'] = X_Y_Spline(temp['year'])
            temp['Data'] = ds
            
            plot_data_new = plot_data_new.append(temp)
            
        plot_data_new.index = list(range(len(plot_data_new)))
        sns.lineplot(ax=axs[r, c], data=plot_data_new, x='year', y='GHG (tCO2e)', hue='Data')
        axs[r, c].set_title(val + ' - ' + country)
fig.tight_layout()
#plt.savefig(plot_filepath + 'Lineplot_CO2_3yrmean_all_GHG.png', dpi=200, bbox_inches='tight')
plt.show()



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

corr_s2 = corr_s
corr_s2['min < 0.3'] = corr_s2.min(1) < 0.3

corr_check = pd.DataFrame(corr_s2.loc[corr_s2['min < 0.3'] == True].drop(['min < 0.3'], axis=1).stack())
corr_check = corr_check.loc[corr_check[0] < 0.3]
corr_check[0] = 1
corr_check = corr_check.unstack().fillna(0)
corr_check_d = corr_check.sum(0)
corr_check_c = corr_check.sum(1)

plot_data = corr_s.stack().reset_index().rename(columns={'level_1':'Data', 0:'Corr'})
sns.boxplot(data=plot_data, x='Data', y='Corr'); plt.show()


temp = plot_data.set_index(['country', 'Data'])
temp['check'] = 0
temp.loc[temp['Corr'] < 0.3, 'check'] = 1
temp = temp[['check']].unstack(level=1).sum(1)

plot_data = plot_data.set_index(['country']).loc[country_order]
plot_data['below 0.3'] = temp > 0

fig, ax = plt.subplots(figsize=(8,5))
sns.boxplot(ax=ax, data=plot_data.reset_index(), x='country', y='Corr', hue='below 0.3'); 
plt.axhline(0, c='k')
plt.xticks(rotation=90); plt.show()

prop_im_ds = summary_im.sum(axis=0, level=0) / summary.sum(axis=0, level=0)
for comb in data_comb:
    d0 = comb.split(', ')[0]
    d1 = comb.split(', ')[1]
    prop_im_ds[comb] = prop_im_ds[[d0, d1]].mean(1)
prop_im_ds = prop_im_ds[data_comb].stack().reset_index().rename(columns={'level_1':'Data', 0:'Prop. imported'})
plot_data = plot_data.merge(prop_im_ds, on=['country', 'Data'])

sns.lmplot(data=plot_data.fillna(0), x='Prop. imported', y='Corr', hue='Data'); plt.show()
sns.lmplot(data=plot_data.fillna(0), y='Prop. imported', x='Corr', hue='Data'); plt.show()

############################
## Checking maxs and mins ##
############################

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