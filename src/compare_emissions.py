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


co2_all = pickle.load(open(emissions_filepath + 'Emissions_aggregated_all.p', 'wb'))
    
years = list(co2_all['exio'].keys())

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

for country in summary.index.levels[0].tolist():
    summary.loc[country,:].plot()
    plt.title(country)
    plt.show()
    
correlation = pd.DataFrame();
for country in summary.index.levels[0].tolist():
    temp = summary.loc[country,:].corr()
    temp['country'] = country
    correlation = correlation.append(temp.set_index('country', append=True))
    
correlation = correlation.stack().reset_index()
correlation['combo'] = correlation['level_0'] + ', ' + correlation['level_2']
keep = ['oecd, figaro', 'oecd, exio', 'oecd, gloria', 
        'figaro, exio', 'figaro, gloria',
        'exio, gloria']
correlation = correlation.loc[correlation['combo'].isin(keep) == True]
correlation = correlation.set_index(['country', 'combo']).rename(columns={0:'corr'})[['corr']]

correlation.unstack(level='combo').plot(kind='bar')
sns.scatterplot(data = correlation.reset_index(), x='combo',y='corr',  hue='country')
sns.boxplot(data = correlation.reset_index().sort_values('corr'), x='combo',y='corr'); plt.xticks(rotation=90)

corr_summary_country = correlation.mean(axis=0, level=0).sort_values('corr', ascending = False)
corr_summary_country.plot(kind='bar')

corr_summary_combo = correlation.mean(axis=0, level=1).sort_values('corr', ascending = False)
corr_summary_combo.plot(kind='bar')
