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

corr_method = 'spearman' # 'pearson

datasets = list(co2_all.keys())
years = list(co2_all[datasets[0]].keys())

data_comb = ['ICIO, Figaro', 'Exiobase, ICIO', 'ICIO, Gloria', 'Exiobase, Figaro', 'Figaro, Gloria', 'Exiobase, Gloria']

country_dict = {'United Kingdom':'UK', 'Czech Republic':'Czechia', 'United States':'USA', 'Rest of the World':'RoW'}
data_dict = {'oecd':'ICIO', 'exio':'Exiobase', 'gloria':'Gloria', 'figaro':'Figaro'}


## IMPORT REG DATA


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
