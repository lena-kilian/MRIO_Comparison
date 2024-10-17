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

error = 'rmse_pct' # 'rmspe'
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
    plt.savefig(plot_filepath + 'Lineplot_CO2_' + country + '_GHG.png', dpi=200, bbox_inches='tight')
    plt.show()
    

for data in ['Total', 'Imports']:
    if data == 'Total':
        temp = cp.copy(summary)
    else:
        temp = cp.copy(summary_im)
    
    for country in temp.index.levels[0]:
        
        plot_data = temp.loc[country].stack().reset_index().rename(columns={'level_1':'Datasets', 0:'tCO2'})
        sns.lmplot(data=plot_data, x='year', y='tCO2', hue='Datasets')
       
        plt.title(country + ' - ' + data)
        plt.show()

# corr with year

check_tot = summary.reset_index().groupby('country').corr().unstack()
check_tot.columns = [x[0] + ', ' + x[1] for x in check_tot.columns.tolist()]
check_tot = check_tot[['year, Exiobase', 'year, ICIO', 'year, Gloria', 'year, Figaro'] + data_comb]


check_imp = summary_im.reset_index().groupby('country').corr().unstack()
check_imp.columns = [x[0] + ', ' + x[1] for x in check_imp.columns.tolist()]
check_imp = check_imp[['year, Exiobase', 'year, ICIO', 'year, Gloria', 'year, Figaro'] + data_comb]


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


      
fig, ax = plt.subplots(figsize=(15, 5))
plot_data = reg_results.set_index('country').loc[country_order].reset_index()
sns.scatterplot(ax=ax, data=reg_results, x='country', y='coef', hue='ds')
ax.axhline(0, c='k')
ax.set_ylabel('Average change per year', fontsize=fs)
ax.set_xlabel('')
plt.xticks(rotation=90, fontsize=fs); 
fig.tight_layout()
plt.savefig(plot_filepath + 'scaterplot_slope_coef_' + country + '_GHG.png', dpi=200, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(15, 5))
plot_data = reg_results.set_index('country').loc[country_order].reset_index()
sns.scatterplot(ax=ax, data=reg_results, x='country', y='coef', hue='ds')
plt.ylim(-20000, 5000)
ax.axhline(0, c='k')
ax.set_ylabel('Average change per year', fontsize=fs)
ax.set_xlabel('')
plt.xticks(rotation=90, fontsize=fs); 
fig.tight_layout()
plt.savefig(plot_filepath + 'scaterplot_slope_coef_' + country + '_COLOUR_ZOOM.png', dpi=200, bbox_inches='tight')
plt.show()


reg_results = reg_results.set_index(['country', 'ds']).unstack()



reg_results2 = cp.copy(reg_results)
for comb in data_comb:
    d0 = comb.split(', ')[0]
    d1 = comb.split(', ')[1]
    reg_results2[('ratio', comb)] = reg_results2[('coef', d0)] / reg_results2[('coef', d1)]
reg_results2 = reg_results2['ratio']

sns.scatterplot(data=reg_results2.stack().reset_index(), x='country', y=0, hue='ds'); 
plt.ylim(-1, 3); plt.show()

sns.boxplot(data=reg_results2.stack().reset_index(), x='ds', y=0, showfliers=False); 
plt.show()

reg_results2_check = reg_results2.stack().reset_index().rename(columns={0:'ratio'})
reg_results2_check = reg_results2_check.loc[reg_results2_check['ratio'] < 0]

reg_results2_check2 = reg_results2_check.set_index(['ds', 'country']).unstack()


reg_results_im = pd.DataFrame()
for country in summary_im.index.levels[0]:
    for ds in ['Exiobase', 'Gloria', 'ICIO', 'Figaro']:
    
        temp = summary_im.loc[country, ds].reset_index()
        
        regressor = LinearRegression()
        regressor.fit(temp[['year']], temp[ds])
        
        new = pd.DataFrame(index=[0])
        new['ds'] = ds
        new['country'] = country
        new['coef'] = regressor.coef_
        
        reg_results_im = reg_results_im.append(new)
reg_results_im = reg_results_im.set_index(['country', 'ds']).unstack()

reg_results_im2 = cp.copy(reg_results_im)
for comb in data_comb:
    d0 = comb.split(', ')[0]
    d1 = comb.split(', ')[1]
    reg_results_im2[('ratio', comb)] = reg_results_im2[('coef', d0)] / reg_results_im2[('coef', d1)]
reg_results_im2 = reg_results_im2['ratio']

sns.scatterplot(data=reg_results_im2.stack().reset_index(), x='country', y=0, hue='ds'); 
plt.ylim(-1, 3); plt.show()

sns.boxplot(data=reg_results_im2.stack().reset_index(), x='ds', y=0, showfliers=False); 
plt.show()

reg_results_im2_check = reg_results_im2.stack().reset_index().rename(columns={0:'ratio'})
reg_results_im2_check = reg_results_im2_check.loc[reg_results_im2_check['ratio'] < 0]

reg_results_im2_check2 = reg_results_im2_check.set'_index(['ds', 'country']).unstack()


'''
'''
#############################
## Change in trend - RMSPE ##
#############################

# Total

temp = summary.unstack('country').swaplevel(axis=1)
data_rmspe = pd.DataFrame(columns=['country'])
# Convert to True vs False
for c in temp.columns.levels[0]:
    temp2 = temp[c]
    for comb in data_comb:
        d0 = comb.split(', ')[0]
        d1 = comb.split(', ')[1]
        temp3 = pd.DataFrame(index=[0])
        temp3['country'] = c
        temp3[comb] = (calc_rmspe(temp2[d0], temp2[d1]) + calc_rmspe(temp2[d1], temp2[d0]))/2
        temp3 = temp3.set_index('country').stack().reset_index().rename(columns={'level_1':'dataset', 0:'RMSPE'})
        data_rmspe = data_rmspe.append(temp3)
data_rmspe = data_rmspe.merge(data_rmspe.groupby('country').mean().reset_index().rename(columns={'RMSPE':'mean'}), on='country').sort_values(['mean', 'dataset'])

# Imports

temp = summary_im.unstack('country').swaplevel(axis=1)
data_rmspe_im = pd.DataFrame(columns=['country'])
# Convert to True vs False
for c in temp.columns.levels[0]:
    temp2 = temp[c]
    for comb in data_comb:
        d0 = comb.split(', ')[0]
        d1 = comb.split(', ')[1]
        temp3 = pd.DataFrame(index=[0])
        temp3['country'] = c
        temp3[comb] = (calc_rmspe(temp2[d0], temp2[d1]) + calc_rmspe(temp2[d1], temp2[d0]))/2
        temp3 = temp3.set_index('country').stack().reset_index().rename(columns={'level_1':'dataset', 0:'RMSPE'})
        data_rmspe_im = data_rmspe_im.append(temp3)
data_rmspe_im = data_rmspe_im.merge(data_rmspe_im.groupby('country').mean().reset_index().rename(columns={'RMSPE':'mean'}), on='country')

# Combine all

data_rmspe = {'Total':data_rmspe, 'Imports':data_rmspe_im}

'''
###################################
## Change in trend - RMSE / Mean ##
###################################

# Total

temp = summary.unstack('country').swaplevel(axis=1)
data_rmse = pd.DataFrame(columns=['country'])
# Convert to True vs False
for c in temp.columns.levels[0]:
    temp2 = temp[c]
    for comb in data_comb:
        d0 = comb.split(', ')[0]
        d1 = comb.split(', ')[1]
        temp3 = pd.DataFrame(index=[0])
        temp3['country'] = c
        temp3['RMSE'] = calc_rmse(temp2[d0], temp2[d1])
        temp3['mean_GHG'] = temp2[[d0, d1]].mean().mean()
        temp3['dataset'] = comb
        data_rmse = data_rmse.append(temp3)
        
        print(c, comb)
       
data_rmse = data_rmse.merge(data_rmse.groupby('country').mean().reset_index().rename(columns={'RMSE':'mean', 'mean_GHG':'country_GHG'}), on='country').sort_values(['mean', 'dataset'])

# Imports

temp = summary_im.unstack('country').swaplevel(axis=1)
data_rmse_im = pd.DataFrame(columns=['country'])
# Convert to True vs False
for c in temp.columns.levels[0]:
    temp2 = temp[c]
    for comb in data_comb:
        d0 = comb.split(', ')[0]
        d1 = comb.split(', ')[1]
        temp3 = pd.DataFrame(index=[0])
        temp3['country'] = c
        temp3['RMSE'] = calc_rmse(temp2[d0], temp2[d1])
        temp3['mean_GHG'] = temp2[[d0, d1]].mean().mean()
        temp3['dataset'] = comb
        data_rmse_im = data_rmse_im.append(temp3)
        data_rmse_im = data_rmse_im.append(temp3)
data_rmse_im = data_rmse_im.merge(data_rmse_im.groupby('country').mean().reset_index().rename(columns={'RMSE':'mean', 'mean_GHG':'country_GHG'}), on='country')

# Combine all

data_rmse_pct = {'Total':data_rmse, 'Imports':data_rmse_im}

data_rmse_pct['Total']['RMSE_PCT'] = data_rmse_pct['Total']['RMSE'] / data_rmse_pct['Total']['mean_GHG'] * 100
data_rmse_pct['Total'] = data_rmse_pct['Total'].set_index(['country', 'dataset'])['RMSE_PCT'].unstack().reset_index()

data_rmse_pct['Imports']['RMSE_PCT'] = data_rmse_pct['Imports']['RMSE'] / data_rmse_pct['Imports']['mean_GHG'] * 100
data_rmse_pct['Imports'] = data_rmse_pct['Imports'].drop_duplicates().set_index(['country', 'dataset'])['RMSE_PCT'].unstack().reset_index()


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

'''
################################
## Average change - direction ##
################################

data_change_direction = {}
for data in ['Total', 'Imports']:
    temp = cp.copy(data_change_og[data])
    
    for comb in data_comb:
        d0 = comb.split(', ')[0]
        d1 = comb.split(', ')[1]
        temp[comb] = 0
        temp.loc[((temp[d0]>1) & (temp[d1]>1) | (temp[d0]<1) & (temp[d1]<1) | 
                  (temp[d0]==1) & (temp[d1]==1)), comb] = 1
        
    temp = temp[data_comb]
        
    data_change_direction[data] = temp.reset_index()
    
'''

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

    
    
'''
#############
## Cluster ##
#############

results = {}

for data in ['Total', 'Imports']:
    results[data] = pd.DataFrame()
    for comp in ['corr', 'rmse_pct', 'change_direction', 'change', 'direction']:
        temp = eval('data_' + comp)[data].set_index('country').loc[order].stack().reset_index().rename(columns={'level_1':'Datasets', 0:'Value', 'dataset':'Datasets'})
        temp['comp'] = comp
        
        results[data] = results[data].append(temp[['country', 'Datasets', 'comp', 'Value']])
    
    check = results[data].set_index(['country', 'Datasets', 'comp']).unstack('comp').droplevel(axis=1, level=0).reset_index()
    sns.scatterplot(data=check, x='corr', y='rmse_pct', hue='Datasets'); plt.show()
    

results_rank = {}

for data in ['Total', 'Imports']:
    results_rank[data] = pd.DataFrame()
    for comp in ['corr', 'rmse_pct']: #, 'change_direction','change', 'direction']:
        temp = eval('data_' + comp)[data].set_index('country').mean(1).reset_index()
        
        if comp in ['rmse_pct']:
            temp['order_asc'] = temp[0]
        if comp in ['change']:
            temp['order_asc'] = temp[0]
            temp.loc[temp['order_asc'] < 0, 'order_asc'] = temp['order_asc'] * -1 + 100
        else:
            temp['order_asc'] = 1/temp[0]
        temp['rank'] = temp['order_asc'].rank()
        
        temp['comp'] = comp
       
        results_rank[data] = results_rank[data].append(temp[['country', 'comp', 'rank']])
    
    results_rank[data] = results_rank[data].set_index(['country', 'comp']).unstack('comp').droplevel(axis=1, level=0)
 
print(results_rank['Total'].corr(method='spearman'))

print(results_rank['Imports'].corr(method='spearman'))


from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

cluster_vars = ['corr', 'rmse_pct', 'direction'] #, 'change', 'change_direction', 

elbow = {}
for data in ['Total', 'Imports']:
    
    temp = cp.copy(results[data].groupby(['country', 'comp']).mean()).unstack('comp').droplevel(axis=1, level=0)[cluster_vars]
    
    for item in temp.columns:
        scaler = MinMaxScaler()
        temp[item] = scaler.fit_transform(temp[item].values.reshape(temp.shape[0],1))
    
    #kmeans = KMeans(init="random", n_clusters=3, n_init=10, max_iter=300, random_state=42)
    
    kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}
   
    sse = []
    n = 20
    for k in range(1, n):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(temp)
        sse.append(kmeans.inertia_)
        
    plt.plot(range(1, n), sse)
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.xticks(range(2, n))
    plt.title(data)
    plt.show()
    
    kl = KneeLocator(range(1, n), sse, curve="convex", direction="decreasing")
    print(kl.elbow)
    
    elbow[data] = kl.elbow
    
    silhouette_coefficients = []
    # Notice you start at 2 clusters for silhouette coefficient
    for k in range(2, n):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(temp)
        score = silhouette_score(temp, kmeans.labels_)
        silhouette_coefficients.append(score)

    plt.style.use("fivethirtyeight")
    plt.plot(range(2, n), silhouette_coefficients)
    plt.xticks(range(2, n))
    plt.title(data)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Coefficient")
    plt.show()
    

clusters = pd.DataFrame()
full = {}
for data in ['Total', 'Imports']:
    temp = cp.copy(results[data].groupby(['country', 'comp']).mean()).unstack('comp').droplevel(axis=1, level=0)[cluster_vars]
    temp_og = cp.copy(temp)
    
    for item in temp.columns:
        scaler = MinMaxScaler()
        temp[item] = scaler.fit_transform(temp[item].values.reshape(temp.shape[0],1))
    
    kmeans = KMeans(n_clusters=elbow[data])
    kmeans.fit(temp)
    
    temp['cluster'] = kmeans.labels_
    temp['cluster'] = ['Group ' + str(x+1) for x in temp['cluster']]
    
    temp = temp_og.join(temp[['cluster']])
    temp['data'] = data
    
    clusters = clusters.append(temp.reset_index()[['country', 'data', 'cluster']])
    
    full[data] = temp
    
    sns.scatterplot(data=temp, x='corr', y='rmse_pct', hue='cluster'); 
    plt.legend(bbox_to_anchor=(1,1))
    plt.title(data); plt.show()
    
    sns.scatterplot(data=temp, x='corr', y='direction', hue='cluster'); 
    plt.legend(bbox_to_anchor=(1,1))
    plt.title(data); plt.show()
    
    sns.scatterplot(data=temp, x='rmse_pct', y='direction', hue='cluster'); 
    plt.legend(bbox_to_anchor=(1,1))
    plt.title(data); plt.show()
    
clusters = clusters.set_index(['country', 'data']).unstack('data')

for data in ['Total', 'Imports']:
    check = full[data].groupby('cluster').mean()
    print(check)
    
'''