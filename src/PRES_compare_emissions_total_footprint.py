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
order_var = 'gdp' # 'prop_order'

# define filepaths
data_filepath = wd + 'ESCoE_Project/data/'
emissions_filepath = wd + 'ESCoE_Project/data/Emissions/'
plot_filepath = wd + 'ESCoE_Project/outputs/compare_all_outputs/plots/'

co2_all = pickle.load(open(emissions_filepath + 'Emissions_aggregated_all.p', 'rb'))

corr_method = 'spearman' # 'pearson

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

# get openness of economy for ordering graphs
# openness = pd.read_excel(data_filepath + 'lookups/lookup_trade_openness.xlsx', sheet_name='agg_data')
# openness = openness.loc[openness['Countries'] != 'ROW Mean'].sort_values('Trade_openness_2018', ascending=False)

# country_order = []
# for c in openness['combined_name']:
#     if c in list(country_dict.keys()):
#         country_order.append(country_dict[c])
#     else:
#         country_order.append(c)

#openness['country'] = country_order

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

####################
## Sort countries ##
####################

# get orders
# get percentage imported
prop_im = pd.DataFrame((summary_im/summary * 100).mean(axis=0, level='country').mean(axis=1)).rename(columns={0:'Percentage CO2 imported'})
# sort by country order
prop_order = prop_im.sort_values('Percentage CO2 imported', ascending=False).index.tolist()

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

# Imports

temp = summary_im.unstack(level=1).stack(level=0)
data_direction_im = temp[years[1:]]
for year in years[1:]:
    data_direction_im[year] =  temp[year - 1] / temp[year]
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

# Combine all

data_direction = {'Total':data_direction['pct_same'].unstack().reset_index(), 
                  'Imports':data_direction_im['pct_same'].unstack().reset_index()}

###################################
## Average change - year on year ##
###################################

# Total

temp = summary.unstack(level=1).stack(level=0)
data_change = temp[years[1:]]
for year in years[1:]:
    
    data_change[year] = ((temp[year] - temp[year - 1]) / temp[[year, year-1]].mean(1)) *100
data_change = data_change.unstack(level=1).stack(level=0)

data_change = data_change.mean(axis=0, level=0)

fig, ax = plt.subplots(figsize=(15, 5))
plot_data = data_change.loc[country_order].stack().reset_index().rename(columns={'level_1':'Dataset', 0:'Mean yearly percentage change'})
sns.scatterplot(ax=ax, data=plot_data, x='country', y='Mean yearly percentage change', hue='Dataset', s=100)
#sns.boxplot(ax=ax, data=plot_data, x='country', y='Mean yearly percentage change', color='white')
sns.pointplot(ax=ax, data=plot_data, x='country', y='Mean yearly percentage change', color='#000000', linestyles="", errorbar='sd', alpha=0.25)
ax.axhline(0, c='k')
ax.set_ylabel('Mean yearly pct. change', fontsize=fs)
ax.set_xlabel('')
plt.xticks(rotation=90, fontsize=fs); 
fig.tight_layout()
plt.savefig(plot_filepath + 'scaterplot_avg_pct_change_Total_GHG.png', dpi=200, bbox_inches='tight')
plt.show()

# Imports

temp = summary_im.unstack(level=1).stack(level=0)
data_change_im = temp[years[1:]]
for year in years[1:]:
    data_change_im[year] = ((temp[year] - temp[year - 1]) / temp[[year, year - 1]].mean(1)) *100
data_change_im = data_change_im.unstack(level=1).stack(level=0)
data_change_im = data_change_im.mean(axis=0, level=0)


fig, ax = plt.subplots(figsize=(15, 5))
plot_data = data_change_im.loc[country_order].stack().reset_index().rename(columns={'level_1':'Dataset', 0:'Mean yearly percentage change'})
sns.scatterplot(ax=ax, data=plot_data, x='country', y='Mean yearly percentage change', hue='Dataset', s=100)
sns.pointplot(ax=ax, data=plot_data, x='country', y='Mean yearly percentage change', color='#000000', linestyles="", errorbar='sd', alpha=0.25)
ax.axhline(0, c='k')
ax.set_ylabel('Mean yearly pct. change', fontsize=fs)
ax.set_xlabel('')
plt.xticks(rotation=90, fontsize=fs); 
fig.tight_layout()
plt.savefig(plot_filepath + 'scaterplot_avg_pct_change_Imports_GHG.png', dpi=200, bbox_inches='tight')
plt.show()


# Combine all

data_change = {'Total':data_change[list(data_dict.values())], 'Imports':data_change_im[list(data_dict.values())]}

#data_change = {'Total':data_change[data_comb].reset_index(), 'Imports':data_change_im[data_comb].reset_index()}

#################
## Correlation ##
#################

# Total

data_corr = summary.reset_index().drop('year', axis=1).groupby('country').corr(method=corr_method).unstack()
data_corr.columns = [x[0] + ', ' + x[1] for x in data_corr.columns]
data_corr = data_corr[data_comb]

# Imports

data_corr_im = summary_im.reset_index().drop('year', axis=1).groupby('country').corr(method=corr_method).unstack()
data_corr_im.columns = [x[0] + ', ' + x[1] for x in data_corr_im.columns]
data_corr_im = data_corr_im[data_comb]
    
# Combine all

data_corr = {'Total':data_corr.reset_index(), 'Imports':data_corr_im.reset_index()}

###################
## Plot together ##
###################

# sort countries by mean_co2
#order = mean_co2['Total'].sort_values('mean_co2', ascending=False).index.tolist()
order = country_order

# Stripplots
fs = 16
pal = 'tab10'
c_box = '#000000'
c_vlines = '#B9B9B9'
point_size = 20

for data in ['Total', 'Imports']:
    for comp in ['rmse_pct', 'direction']:
        plot_data = eval('data_' + comp)[data].set_index('country').loc[order].stack().reset_index().rename(columns={'level_1':'Datasets', 0:'Value'})
        temp = plot_data.groupby(['country']).describe().stack(level=1).loc[order].reset_index().rename(columns={'level_1':'stat'})
        temp = temp.loc[temp['stat'].isin(['min', 'max']) == True]
        
        fig, ax = plt.subplots(figsize=(20, 5), sharex=True)
    
        sns.scatterplot(ax=ax, data=temp, x='country', y='Value', color='#000000', s=150, marker='_')
        sns.pointplot(ax=ax, data=plot_data, x='country', y='Value', color='#000000', linestyles="", errorbar='sd')
        ax.set_xlabel('')
        ax.set_ylabel(comp, fontsize=fs)
        ax.tick_params(axis='y', labelsize=fs)
        #ax.set_yscale('log')
        
        if comp =='change':
            ax.set_ylim(-10, 10)
        
      
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=fs); 
    
        for c in range(len(plot_data['country'].unique())):
            ax.axvline(c+0.5, c=c_vlines, linestyle=':')
            
        ax.axhline(0, c=c_vlines)
    
        fig.tight_layout()
        plt.savefig(plot_filepath + 'Pointplot_similarity_bycountry_' + data + '_' + comp + '_GHG.png', dpi=200, bbox_inches='tight')
        plt.show()
        
for comp in ['direction', 'rmse_pct']:
    plot_data = eval('data_' + comp)['Total'].set_index('country').stack().reset_index().rename(columns={'level_1':'Datasets', 0:'Value', 'dataset':'Datasets'})
    plot_data['Type'] = 'Total'
    temp = eval('data_' + comp)['Imports'].set_index('country').stack().reset_index().rename(columns={'level_1':'Datasets', 0:'Value', 'dataset':'Datasets'})
    temp['Type'] = 'Imports'
    
    plot_data = plot_data.append(temp)
    
    fig, ax = plt.subplots(figsize=(20, 5), sharex=True)

    sns.boxplot(ax=ax, data=plot_data, x='Datasets', y='Value', hue='Type', showfliers=False)
    ax.set_xlabel('')
    ax.set_ylabel(comp, fontsize=fs)
    ax.tick_params(axis='y', labelsize=fs)
    #ax.set_yscale('log')
  
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=fs); 

    for c in range(len(plot_data['Datasets'].unique())):
        ax.axvline(c+0.5, c=c_vlines, linestyle=':')
        
    ax.axhline(0, c=c_vlines)

    fig.tight_layout()
    plt.savefig(plot_filepath + 'Boxplot_similarity_bydata_' + comp + '_GHG.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    
for data in ['Total', 'Imports']:
    
    plot_data = data_change[data].loc[order].stack().reset_index().rename(columns={'level_1':'Datasets', 0:'Value'})
    temp = plot_data.groupby(['country']).describe().stack(level=1).loc[order].reset_index().rename(columns={'level_1':'stat'})
    temp = temp.loc[temp['stat'].isin(['min', 'max']) == True]
    
    fig, ax = plt.subplots(figsize=(20, 5), sharex=True)
    
    #sns.scatterplot(ax=ax, data=temp, x='country', y='Value', color='#000000', s=150, marker='_')
    sns.pointplot(ax=ax, data=plot_data, x='country', y='Value', color='#000000', linestyles="", errorbar='sd', alpha=0.5,)
    sns.scatterplot(ax=ax, data=plot_data, x='country', y='Value', hue='Datasets', s=150)
    ax.set_xlabel('')
    ax.set_ylabel('Mean annual percentage change', fontsize=fs)
    ax.tick_params(axis='y', labelsize=fs)
    #ax.set_yscale('log')
  
    plt.xticks(rotation=90, fontsize=fs); 

    for c in range(len(plot_data['country'].unique())):
        ax.axvline(c+0.5, c=c_vlines, linestyle=':')
        
    ax.axhline(0, c=c_vlines)

    fig.tight_layout()
    plt.savefig(plot_filepath + 'Pointplot_similarity_bycountry_' + data + '_averagepctchange_GHG.png', dpi=200, bbox_inches='tight')
    plt.show()
    


for data in ['Total', 'Imports']:
    check = data_change[data]
    check['Same'] = 0
    check.loc[(check['ICIO'] >= 0) & 
              (check['Exiobase'] >= 0) &
              (check['Figaro'] >= 0) &
              (check['Gloria'] >= 0), 'Same'] = 1
    
    check.loc[(check['ICIO'] <= 0) & 
              (check['Exiobase'] <= 0) &
              (check['Figaro'] <= 0) &
              (check['Gloria'] <= 0), 'Same'] = 1
    
    check = check.loc[country_order]
    print(data, 44-check['Same'].sum())

    