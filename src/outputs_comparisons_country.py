# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:13:18 2024

@author: geolki
"""

import pandas as pd
from sys import platform
import pickle
import numpy as np
import copy as cp
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
outputs_filepath = wd + 'ESCoE_Project/outputs/compare_all_outputs/'
plot_filepath = outputs_filepath + 'plots/'

order_var = 'gdp' # 'prop_order'
same_direction_pct_cutoff = 1

# import data
summary_co2 = pickle.load(open(outputs_filepath + 'summary_co2_country.p', 'rb'))
mean_co2 = pickle.load(open(outputs_filepath + 'mean_co2_country.p', 'rb'))
data_rmse_pct = pickle.load(open(outputs_filepath + 'rmse_pct_country.p', 'rb'))
data_direction = pickle.load(open(outputs_filepath + 'direction_annual_country.p', 'rb'))
reg_results = pickle.load(open(outputs_filepath + 'regression_country.p', 'rb'))

country_order = pickle.load(open(outputs_filepath + 'country_order.p', 'rb'))[order_var]
datasets = summary_co2['Total'].columns.tolist(); datasets.sort()
years = summary_co2['Total'].index.levels[0].tolist()

data_comb = ['ICIO, Figaro', 'Exiobase, ICIO', 'ICIO, Gloria', 'Exiobase, Figaro', 'Figaro, Gloria', 'Exiobase, Gloria']

###############
## Summarise ##
###############


# Total
summary = summary_co2['Total']

# Imports
summary_im = summary_co2['Imports']

###################################
## Change in trend - RMSE / Mean ##
###################################


#################################
## Change in trend - Direction ##
#################################





###################################
## Regress footprints over years ##
###################################

for item in ['Total', 'Imports']:
    plot_data = reg_results[item].drop('reg_validation_pct', axis=1)
    plot_data.loc[(plot_data['max'] <= same_direction_pct_cutoff) & (plot_data['min'] >= same_direction_pct_cutoff * -1), 'Same direction'] = True
    plot_data = plot_data.loc[country_order].set_index('Same direction', append=True)\
        .stack().reset_index().rename(columns={0:'Average pct change', 'level_2':'Data'})
    plot_data['Same direction'] = pd.Categorical(plot_data['Same direction'], categories=[True, False], ordered=True)
        
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.scatterplot(ax=ax, data=plot_data, x='country', y='Average pct change', style='Data', hue='Same direction'); 
    plt.xticks(rotation=90); plt.title(item)
    plt.axhline(same_direction_pct_cutoff, linestyle=':', c='k'); plt.axhline(same_direction_pct_cutoff *-1, linestyle=':', c='k'); 
    plt.axhline(0, c='k');
    plt.legend(bbox_to_anchor=(1,1))
    plt.xlabel('')
    
    fig.tight_layout()
    plt.savefig(plot_filepath + 'scatterplot_regresults_bycountry_' + item + '_GHG.png', dpi=200, bbox_inches='tight')
    plt.show()
