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
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures

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

country_dict = {'United Kingdom':'UK', 'Czech Republic':'Czechia', 'United States':'USA', 'Rest of the World':'RoW'}
data_dict = {'oecd':'ICIO', 'exio':'Exiobase', 'gloria':'Gloria', 'figaro':'Figaro'}

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

###################################
## Regress footprints over years ##
###################################

# Total emissions
# test which fit is best linear, quadratic, cubic

reg_fit = pd.DataFrame()
for country in summary.index.levels[0]:
    for ds in ['Exiobase', 'Gloria', 'ICIO', 'Figaro']:
    
        temp = summary.loc[country, ds].reset_index()
        #temp = temp.loc[temp['year'] != year]
        
        y = temp[ds]
        x = temp[['year']]
        
        x1 = sm.add_constant(x)
        
        polynomial_features= PolynomialFeatures(degree=2)
        x2 = polynomial_features.fit_transform(x)
        
        polynomial_features= PolynomialFeatures(degree=3)
        x3 = polynomial_features.fit_transform(x)
 
        #fit regression model
        model1 = sm.OLS(y, x1).fit()
        model2 = sm.OLS(y, x2).fit()
        model3 = sm.OLS(y, x3).fit()
        
        new = pd.DataFrame(index=[0])
        new['ds'] = ds
        new['country'] = country
        new['aic1'] = model1.aic
        new['aic2'] = model2.aic
        new['aic3'] = model3.aic
        
        reg_fit = reg_fit.append(new)
reg_fit = reg_fit.set_index(['country', 'ds'])
reg_fit['aic_prop2'] = reg_fit['aic2'] / reg_fit['aic1']
reg_fit['aic_prop3'] = reg_fit['aic3'] / reg_fit['aic1']

# Sense check with individual years removed

reg_check = pd.DataFrame()
for year in years + [0]:
    for country in summary.index.levels[0]:
        for ds in ['Exiobase', 'Gloria', 'ICIO', 'Figaro']:
        
            temp = summary.loc[country, ds].reset_index()
            temp = temp.loc[temp['year'] != year]
            
            y = temp[ds]
            x = temp[['year']]
            x = sm.add_constant(x)
 
            #fit regression model
            model = sm.OLS(y, x).fit()
            
            new = pd.DataFrame(index=[0])
            new['ds'] = ds
            new['country'] = country
            new['coef'] = model.params['year']
            new['r2'] = model.rsquared
            new['r2_ajd'] = model.rsquared_adj
            new['year'] = year
                        
            
            reg_check = reg_check.append(new)

reg_check_summary = reg_check.loc[reg_check['year'] != 0].drop(['year', 'r2_ajd'], axis=1)\
    .groupby(['country', 'ds']).describe().swaplevel(axis=1)[['mean', 'std']]

reg_check2 = reg_check.groupby(['country', 'year']).describe()['coef'][['min', 'max']]
reg_check2['crosses 0'] = 0
reg_check2.loc[(reg_check2['max'] > 0) & (reg_check2['min'] < 0), 'crosses 0'] = 1

reg_check2 = reg_check2[['crosses 0']].unstack()


# All years

reg_results = reg_check.loc[reg_check['year'] == 0][['ds', 'country', 'coef']]
reg_results = reg_results.set_index(['country', 'ds']).unstack().droplevel(axis=1, level=0)
reg_results = reg_results.loc[country_order]

reg_result2 = cp.copy(reg_results)
reg_result2['mean_co2'] = summary.reset_index('country').groupby('country').mean().mean(1)
reg_result2 = reg_result2.apply(lambda x: x/reg_result2['mean_co2'] *100)

temp = reg_result2.drop('mean_co2', axis=1).T.describe().T[['max', 'min']]
temp['crosses 0'] = False
temp.loc[(temp['max'] > 0) & (temp['min'] < 0), 'crosses 0'] = True

# plot
plot_data = reg_result2.drop('mean_co2', axis=1).join(temp[['crosses 0']]).loc[country_order].set_index('crosses 0', append=True)\
    .stack().reset_index().rename(columns={0:'Average pct change', 'level_2':'Data'})
    
fig, ax = plt.subplots(figsize=(15, 5))
sns.scatterplot(ax=ax, data=plot_data, x='country', y='Average pct change', style='Data', hue='crosses 0'); 
plt.axhline(-1, linestyle=':', c='k'); plt.axhline(1, linestyle=':', c='k'); 
plt.xticks(rotation=90); plt.title('Total')
plt.axhline(0, c='k');

fig.tight_layout()
plt.savefig(plot_filepath + 'scatterplot_regresults_bycountry_total_GHG.png', dpi=200, bbox_inches='tight')
plt.show()


# Imported emissions

# test which fit is best linear, quadratic, cubic

reg_fit_im = pd.DataFrame()
for country in summary_im.index.levels[0]:
    for ds in ['Exiobase', 'Gloria', 'ICIO', 'Figaro']:
    
        temp = summary_im.loc[country, ds].reset_index()
        #temp = temp.loc[temp['year'] != year]
        
        y = temp[ds]
        x = temp[['year']]
        
        x1 = sm.add_constant(x)
        
        polynomial_features= PolynomialFeatures(degree=2)
        x2 = polynomial_features.fit_transform(x)
        
        polynomial_features= PolynomialFeatures(degree=3)
        x3 = polynomial_features.fit_transform(x)
 
        #fit regression model
        model1 = sm.OLS(y, x1).fit()
        model2 = sm.OLS(y, x2).fit()
        model3 = sm.OLS(y, x3).fit()
        
        new = pd.DataFrame(index=[0])
        new['ds'] = ds
        new['country'] = country
        new['aic1'] = model1.aic
        new['aic2'] = model2.aic
        new['aic3'] = model3.aic
        
        reg_fit_im = reg_fit_im.append(new)
reg_fit_im = reg_fit_im.set_index(['country', 'ds'])
reg_fit_im['aic_prop2'] = reg_fit_im['aic2'] / reg_fit_im['aic1']
reg_fit_im['aic_prop3'] = reg_fit_im['aic3'] / reg_fit_im['aic1']

# Sense check with individual years removed

reg_check_im = pd.DataFrame()
for year in years + [0]:
    for country in summary_im.index.levels[0]:
        for ds in ['Exiobase', 'Gloria', 'ICIO', 'Figaro']:
        
            temp = summary_im.loc[country, ds].reset_index()
            temp = temp.loc[temp['year'] != year]
            
            y = temp[ds]
            x = temp[['year']]
            x = sm.add_constant(x)
            
            #fit regression model
            model = sm.OLS(y, x).fit()
            
            new = pd.DataFrame(index=[0])
            new['ds'] = ds
            new['country'] = country
            new['coef'] = model.params['year']
            new['r2'] = model.rsquared
            new['r2_ajd'] = model.rsquared_adj
            new['year'] = year
            
            reg_check_im = reg_check_im.append(new)

reg_check_im2 = reg_check_im.groupby(['country', 'year']).describe()['coef'][['min', 'max']]
reg_check_im2['crosses 0'] = 0
reg_check_im2.loc[(reg_check_im2['max'] > 0) & (reg_check_im2['min'] < 0), 'crosses 0'] = 1

reg_check_im2 = reg_check_im2[['crosses 0']].unstack()


# All years

reg_results_im = reg_check_im.loc[reg_check_im['year'] == 0][['ds', 'country', 'coef']]
reg_results_im = reg_results_im.set_index(['country', 'ds']).unstack().droplevel(axis=1, level=0)

reg_result2_im = cp.copy(reg_results_im)
reg_result2_im['mean_co2'] = summary_im.reset_index('country').groupby('country').mean().mean(1)
reg_result2_im = reg_result2_im.apply(lambda x: x/reg_result2_im['mean_co2'] *100)

temp = reg_result2_im.drop('mean_co2', axis=1).T.describe().T[['max', 'min']]
temp['crosses 0'] = False
temp.loc[(temp['max'] > 0) & (temp['min'] < 0), 'crosses 0'] = True

# plot
plot_data = reg_result2_im.drop('mean_co2', axis=1).join(temp[['crosses 0']]).loc[country_order].set_index('crosses 0', append=True)\
    .stack().reset_index().rename(columns={0:'Average pct change', 'level_2':'Data'})
    
fig, ax = plt.subplots(figsize=(15, 5))
sns.scatterplot(ax=ax, data=plot_data, x='country', y='Average pct change', style='Data', hue='crosses 0'); 
plt.xticks(rotation=90); plt.title('Imports')
plt.axhline(-1, linestyle=':', c='k'); plt.axhline(1, linestyle=':', c='k'); 
plt.axhline(0, c='k');

fig.tight_layout()
plt.savefig(plot_filepath + 'scatterplot_regresults_bycountry_imports_GHG.png', dpi=200, bbox_inches='tight')
plt.show()
