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
plot_filepath = wd + 'ESCoE_Project/outputs/compare_all_outputs/plots/'


co2_all = pickle.load(open(emissions_filepath + 'Emissions_aggregated_all.p', 'rb'))
    

datasets = list(co2_all.keys())
years = list(co2_all[datasets[0]].keys())

data_comb = ['oecd, figaro', 'oecd, exio', 'oecd, gloria', 
             'figaro, exio', 'figaro, gloria',
             'exio, gloria']


###############
## Summarise ##
###############

summary = pd.DataFrame()
for year in years:
    temp_oecd = pd.DataFrame(co2_all['oecd'][year].sum(axis=0, level=1).sum(axis=1)).rename(columns={0:'oecd'})
    temp_figaro = pd.DataFrame(co2_all['figaro'][year].sum(axis=0, level=1).sum(axis=1)).rename(columns={0:'figaro'})
    temp_gloria = pd.DataFrame(co2_all['gloria'][year].sum(axis=0, level=1).sum(axis=1)).rename(columns={0:'gloria'})
    temp_exio = pd.DataFrame(co2_all['exio'][year].sum(axis=0, level=1).sum(axis=1)).rename(columns={0:'exio'})
    
    temp = temp_oecd.join(temp_figaro).join(temp_gloria).join(temp_exio)
    
    temp['year'] = year
    
    summary = summary.append(temp.reset_index())
    
summary = summary.rename(columns={'index':'sector'}).set_index(['sector', 'year']).fillna(0)

#################
## Correlation ##
#################
    
correlation = pd.DataFrame();
for sector in summary.index.levels[0].tolist():
    temp = summary.loc[sector,:].corr()
    temp['sector'] = sector
    correlation = correlation.append(temp.set_index('sector', append=True))
    
correlation = correlation.stack().reset_index()
correlation['combo'] = correlation['level_0'] + ', ' + correlation['level_2']
correlation = correlation.loc[correlation['combo'].isin(data_comb) == True]
correlation = correlation.set_index(['sector', 'combo']).rename(columns={0:'corr'})[['corr']]

correlation.unstack('combo').plot(kind='bar')
sns.scatterplot(data = correlation.reset_index(), x='combo',y='corr',  hue='sector'); plt.legend(bbox_to_anchor=(1,1)); plt.show()
sns.boxplot(data = correlation.reset_index().sort_values('corr'), x='combo',y='corr'); plt.xticks(rotation=90)
plt.savefig(plot_filepath + 'boxplot_sector_correlation.png', dpi=200, bbox_inches='tight')


correlation = correlation.unstack('combo')
correlation[('dataset', 'mean')] = correlation.mean(1)
correlation = correlation.append(pd.DataFrame(correlation.mean(0)).T.rename(index={0:'mean'}))
correlation = correlation.sort_values(('dataset', 'mean'), ascending=False).T.sort_values('mean', ascending=False).T

corr_summary_sector = correlation[['corr']].stack(level=1).mean(axis=0, level=0).sort_values('corr', ascending = False)
corr_summary_sector.plot(kind='bar')
plt.savefig(plot_filepath + 'barplot_sector_correlation_mean_bysector.png', dpi=200, bbox_inches='tight')

corr_summary_combo = correlation[['corr']].stack(level=1).mean(axis=0, level=1).sort_values('corr', ascending = False)
corr_summary_combo.plot(kind='bar')
plt.savefig(plot_filepath + 'barplot_sector_correlation_mean_bydatapair.png', dpi=200, bbox_inches='tight')