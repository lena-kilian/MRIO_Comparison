# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 10:30:46 2023

@author: geolki
"""

import pandas as pd
from sys import platform
import pickle
import numpy as np
import copy as cp
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures

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

co2_all = pickle.load(open(emissions_filepath + 'Emissions_aggregated_all.p', 'rb'))

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

summary_co2 = {'Total':summary, 'Imports':summary_im}
# save
pickle.dump(summary_co2, open(outputs_filepath + 'summary_co2_country.p', 'wb'))

# Get means
mean_co2 = {'Total' : pd.DataFrame(summary.mean(axis=0, level='country').mean(axis=1)).rename(columns={0:'mean_co2'}), 
            'Imports' : pd.DataFrame(summary_im.mean(axis=0, level='country').mean(axis=1)).rename(columns={0:'mean_co2'})}
# save
pickle.dump(mean_co2, open(outputs_filepath + 'mean_co2_country.p', 'wb'))

####################
## Sort countries ##
####################

# get orders
# percentage imported
prop_im = (summary_im / summary * 100).mean(axis=0, level='country').mean(axis=1).reset_index().rename(columns={0:'prop_imported'})
prop_order = prop_im.sort_values('prop_imported', ascending=False).set_index('country').rename(index=country_dict).index.tolist()

# GDP
country_list = pd.DataFrame(co2_all[datasets[0]][years[0]].index.levels[0]).set_index(0)
country_list['index_test'] = 1
# GDP data https://data.worldbank.org/indicator/NY.GDP.MKTP.CD
gdp = pd.DataFrame(pd.read_csv(data_filepath + 'GDP/GDP.csv', header=2, index_col=0)[[str(x) for x in years]].mean(1)).dropna(0)\
    .rename(index={'Korea, Rep.':'South Korea', 'Slovak Republic':'Slovakia', 'Czechia':'Czech Republic', 'Russian Federation':'Russia', 'Turkiye':'Turkey'})
gdp = gdp.join(country_list, how='outer')
gdp['country'] = gdp.index.tolist()
gdp.loc[gdp['index_test'] != 1, 'country'] = 'Rest of the World'
gdp = gdp.groupby('country').sum()[0].sort_values(0, ascending=False).rename(index=country_dict).index.tolist()

# economic openness
openness = pd.read_excel(data_filepath + 'lookups/lookup_trade_openness.xlsx', sheet_name='agg_data')
openness = openness.loc[openness['Countries'] != 'ROW Mean'].sort_values('Trade_openness_2018', ascending=False)\
    .set_index('combined_name').rename(index=country_dict).index.tolist()

# Combine and save
country_order = {'gdp':gdp, 'prop_imports':prop_order, 'openness':openness}
pickle.dump(country_order, open(outputs_filepath + 'country_order.p', 'wb'))

###################################
## Change in trend - RMSE / Mean ##
###################################

def calc_rmse(x1, x2):
    diff = x1-x2
    diff_sq = diff**2
    mean_sq = np.mean(diff_sq)
    error = np.sqrt(mean_sq)
    return(error)

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

# save
pickle.dump(data_rmse_pct, open(outputs_filepath + 'rmse_pct_country.p', 'wb'))

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
data_direction = data_direction.reset_index().merge(prop_im, on='country')

# Imports
temp = summary_im.unstack(level=1).stack(level=0)
data_direction_im = temp[years[1:]]
for year in years[1:]:
    data_direction_im[year] = temp[year] / temp[year - 1]
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
data_direction_im = data_direction_im.reset_index().merge(prop_im, on='country')

# Combine all
data_direction = {'Total':data_direction, 'Imports':data_direction_im}

# save
pickle.dump(data_direction, open(outputs_filepath + 'direction_annual_country.p', 'wb'))


###################################
## Regress footprints over years ##
###################################

for item in ['Total', 'Imports']:
    # Total emissions
    # test which fit is best linear, quadratic, cubic
    reg_fit = pd.DataFrame()
    for country in summary_co2[item].index.levels[0]:
        for ds in ['Exiobase', 'Gloria', 'ICIO', 'Figaro']:
        
            temp = summary_co2[item].loc[country, ds].reset_index()
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
    
    reg_fit.to_csv(outputs_filepath + 'regression_country_linieraty_AIC_' + item + '.csv')
    

regression_results = {}
for item in ['Total', 'Imports']:
    reg_check = pd.DataFrame()
    # Sense check with individual years removed
    for year in years + [0]:
        for country in summary_co2[item].index.levels[0]:
            for ds in ['Exiobase', 'Gloria', 'ICIO', 'Figaro']:
            
                temp = summary_co2[item].loc[country, ds].reset_index()
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
    
    reg_check2 = reg_check.groupby(['country', 'year']).describe()['coef'][['min', 'max']]
    reg_check2['Same direction'] = 1
    reg_check2.loc[(reg_check2['max'] > 0) & (reg_check2['min'] < 0), 'Same direction'] = 0
    reg_check2 = reg_check2[['Same direction']].unstack()['Same direction']
    reg_check2['reg_validation_pct'] = np.abs(reg_check2[0] - reg_check2[years].mean(1)) * 100
    
    # All years
    reg_results = reg_check.loc[reg_check['year'] == 0][['ds', 'country', 'coef']]
    reg_results = reg_results.set_index(['country', 'ds']).unstack().droplevel(axis=1, level=0)
    
    reg_result2 = cp.copy(reg_results)
    reg_result2['mean_co2'] = summary.reset_index('country').groupby('country').mean().mean(1)
    reg_result2 = reg_result2.apply(lambda x: x/reg_result2['mean_co2'] *100).drop('mean_co2', axis=1)
    
    temp = reg_result2.T.describe().T[['max', 'min']]
    temp['Same direction'] = True
    temp.loc[(temp['max'] > 0) & (temp['min'] < 0), 'Same direction'] = False
    
    regression_results[item] = reg_result2.join(temp).join(reg_check2[['reg_validation_pct']])

# save
pickle.dump(regression_results, open(outputs_filepath + 'regression_country.p', 'wb'))
