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

ind_dict = {'Electricity, gas, steam and air conditioning supply':'Electricity &\ngas',
            'Mining and quarrying, energy producing products':'Mining',
            'Agriculture, hunting, forestry':'Agriculture &\nforestry',
            'Chemical, parmaceuticals and botanical products':'Chemicals &\nparmaceuticals', 
            'Basic metals':'Basic metals',
            'Machinery, computer, electronic, optical equipment, and other machinery and equipment':'Machinery &\nequipment',
            'Water transport':'Water\ntransport', 
            'Food products, beverages and tobacco':'Processing\nof food &\nbeverages',
            'Coke and refined petroleum products':'Coke &\nrefined\npetroleum',
            'Other non-metallic mineral products':'Other non-\nmetallic\nminerals',
            'Manufacturing nec; repair and installation of machinery and equipment':'Repair &\nmanufacturing nec',
            'Water supply; sewerage, waste management and remediation activities':'Water supply &\nwaste\nmanagement',
            'Land transport and transport via pipelines':'Land &\npipeline\ntransport',
            'Textiles, textile products, leather and footwear':'Textiles',
            'Air transport':'Air transport', 
            'Motor vehicles, trailers and semi-trailers':'Motor vehicles',
            'Wholesale and retail trade; repair of motor vehicles':'Wholesale and retail',
            'Administrative, support and other professional and supporting transport services':'Professional &\nsupporting transport services',
            'Other transport equipment':'Other transport equipment',
            'Accommodation and food service activities':'Accommodation &\nfood services',
            'Paper products and printing':'Paper and printing', 
            'Fabricated metal products':'Fabricated metals',
            'Rubber and plastics products':'Rubber &\nplastics',
            'IT, information, postal, communication services and publishing':'IT, communication services &\npublishing',
            'Construction':'Construction', 
            'Financial and insurance activities':'Finance &\ninsurance',
            'Wood and products of wood and cork':'Wood &\ncork', 
            'Other service activities':'Other services',
            'Public administration and defence; compulsory social security':'Public administration &\ndefence',
            'Activities of households as employers; undifferentiated goods- and services-producing activities of households for own use':'Households as employers',
            'Real estate activities':'Real estate services', 
            'Fishing and aquaculture':'Fishing &\naquaculture', 
            'Education':'Education',
            'Human health and social work activities':'Human health &\nsocial work',
            'Activities of extraterritorial organisations and bodies':'Extraterritorial organisations',
            'Private households':'Private households'
            }


###############
## Summarise ##
###############

# Total

summary = pd.DataFrame()
for year in years:
    temp_oecd = pd.DataFrame(co2_all['oecd'][year].sum(axis=1, level=0).sum(axis=0, level=1).stack()).rename(columns={0:'oecd'})
    temp_figaro = pd.DataFrame(co2_all['figaro'][year].sum(axis=1, level=0).sum(axis=0, level=1).stack()).rename(columns={0:'figaro'})
    temp_gloria = pd.DataFrame(co2_all['gloria'][year].sum(axis=1, level=0).sum(axis=0, level=1).stack()).rename(columns={0:'gloria'})
    temp_exio = pd.DataFrame(co2_all['exio'][year].sum(axis=1, level=0).sum(axis=0, level=1).stack()).rename(columns={0:'exio'})
    # merge all
    temp = temp_oecd.join(temp_figaro, how='outer').join(temp_gloria, how='outer').join(temp_exio, how='outer').fillna(0)
    temp['year'] = year
    summary = summary.append(temp.reset_index())
summary = summary.rename(columns={'level_0':'industry', 'level_1':'country'}).set_index(['country', 'industry'])\
    .rename(index=country_dict).swaplevel(axis=0).rename(index=ind_dict).set_index('year', append=True).rename(columns=data_dict)
    
# Imports

summary_im = pd.DataFrame()
for year in years:
    temp = {}
    for item in ['oecd', 'figaro', 'gloria', 'exio']:
        temp[item] = co2_all[item][year]
        for country in temp[item].index.levels[0]:
            temp[item].loc[country, country] = 0
        temp[item] = pd.DataFrame(temp[item].sum(axis=1, level=0).sum(axis=0, level=1).stack()).rename(columns={0:item})
    # merge all
    temp_all = temp['oecd'].join(temp['figaro']).join(temp['gloria']).join(temp['exio'])
    temp_all['year'] = year
    summary_im = summary_im.append(temp_all.reset_index())
summary_im = summary_im.rename(columns={'level_0':'industry', 'level_1':'country'}).set_index(['country', 'industry'])\
    .rename(index=country_dict).swaplevel(axis=0).rename(index=ind_dict).set_index('year', append=True).rename(columns=data_dict)
    

# get percentage imported
prop_im = pd.DataFrame((summary_im/summary * 100).mean(axis=0, level='country').mean(axis=1)).rename(columns={0:'Percentage CO2 imported'})

country_order = prop_im.sort_values('Percentage CO2 imported', ascending=False).index.tolist()

################
## Rank plots ##
################

# totals
ax_max = summary.max().max()
cols = summary.columns.tolist()
for i in range(len(cols)-1):
    for j in range(i+1, len(cols)):
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.scatterplot(ax=ax, data=summary, x=cols[i], y=cols[j], s=2)
        plt.xlim(1, ax_max); plt.ylim(1, ax_max)
        plt.xscale('log'); plt.yscale('log')
        plt.show()
    
# imports 
ax_max = summary_im.max().max()
cols = summary_im.columns.tolist()
for i in range(len(cols)-1):
    for j in range(i+1, len(cols)):
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.scatterplot(ax=ax, data=summary_im, x=cols[i], y=cols[j], s=2)
        plt.xlim(1, ax_max); plt.ylim(1, ax_max)
        plt.xscale('log'); plt.yscale('log')
        plt.show()
    

###################
## Industry Corr ##
###################

# get mean emissions by sector and country
# total
means = summary.mean(axis=0, level=['country', 'industry'])
corr = means.reset_index().groupby(['country']).corr(method='spearman').unstack(level=1)
corr.columns = [x[0] + ', ' + x[1] for x in corr.columns.tolist()]
corr = corr[data_comb]

corr = corr.stack().reset_index().rename(columns={'level_1':'Data', 0:'Total'})

# imports
means_im = summary_im.mean(axis=0, level=['country', 'industry'])
corr_im = means_im.reset_index().groupby(['country']).corr(method='spearman').unstack(level=1)
corr_im.columns = [x[0] + ', ' + x[1] for x in corr_im.columns.tolist()]
corr_im = corr_im[data_comb]

corr_im = corr_im.stack().reset_index().rename(columns={'level_1':'Data', 0:'Imports'})
    
# Histogram
corr_all = corr.merge(corr_im, on =['country', 'Data']).set_index(['country', 'Data']).stack().reset_index().rename(columns={'level_2':'Type', 0:"Spearman's Rho"})

fig, axs = plt.subplots(nrows=len(data_comb), figsize=(10, 10), sharex=True)#, sharey=True)
for i in range(len(data_comb)):
    item = data_comb[i]
    plot_data = corr_all.loc[corr_all['Data'] == item]
    sns.histplot(ax=axs[i], data=plot_data, x="Spearman's Rho", hue='Type', binwidth=0.025)
    #sns.kdeplot(ax=axs[i], data=plot_data, x="Spearman's Rho", hue='Type')
    axs[i].set_title(item)
    axs[i].set_xlim(0, 1)
    #axs[i].set_xlim(-1, 1)
fig.tight_layout()
plt.savefig(plot_filepath + 'histplot_CO2_sector_corr_by_data_GHG.png', dpi=200, bbox_inches='tight')
plt.show() 

# Scatterplot
plot_data = corr.set_index('Data').loc[data_comb].reset_index().rename(columns={'Total':"Spearman's Rho"})
plot_data_im = corr_im.set_index('Data').loc[data_comb].reset_index().rename(columns={'Imports':"Spearman's Rho"})
plot_data['Country'] = '                     ' + plot_data['country']
plot_data_im['Country'] = '                     ' + plot_data_im['country']

fig, axs = plt.subplots(nrows=2, figsize=(20, 10), sharex=True)
# total
sns.scatterplot(ax=axs[0], data=plot_data, x='country', y="Spearman's Rho", hue='Data', s=120)
# imports
sns.scatterplot(ax=axs[1], data=plot_data_im, x='country', y="Spearman's Rho", hue='Data',  s=120, legend=False)

axs[1].xaxis.set_ticks_position('top') # the rest is the same
axs[1].set_xticklabels(plot_data['Country'].unique(), rotation=90, va='center', fontsize=fs); 
axs[0].legend(bbox_to_anchor=(1,1))

for i in [0.3, 0.5, 0.7]:
    axs[0].axhline(i, c='k', linestyle=':')
    axs[1].axhline(i, c='k', linestyle=':')
    
fig.tight_layout()
plt.savefig(plot_filepath + 'scatterplot_CO2_sector_corr_by_country_GHG.png', dpi=200, bbox_inches='tight')
plt.show()

####################
## Industry Top 5 ##
####################

# get mean emissions by sector and country
n = 11

# total
sums = summary.sum(axis=0, level=['industry', 'year']).unstack('year').T
sums['Total'] = sums.sum(1)
sums = sums.T.stack('year')

order = pd.DataFrame(sums.mean(axis=0, level='industry').mean(1)).sort_values(0, ascending=False)
order['cumsum'] = order[0].cumsum()
order['cumpct'] = order['cumsum'] / order[0].sum() * 100
order_list = order.iloc[:n].index.tolist()

sums = pd.DataFrame(sums.stack())

#imports
sums_im = summary_im.sum(axis=0, level=['industry', 'year']).unstack('year').T
sums_im['Total'] = sums_im.sum(1)
sums_im = sums_im.T.stack('year')
order_im = pd.DataFrame(sums_im.mean(axis=0, level='industry').mean(1)).sort_values(0, ascending=False)
order_im['cumsum'] = order_im[0].cumsum()
order_im['cumpct'] = order_im['cumsum'] / order_im[0].sum() * 100
order_im_list = order_im.iloc[:n].index.tolist()

sums_im = pd.DataFrame(sums_im.stack())


#plot
fig, axs = plt.subplots(figsize=(17, 10), ncols=2)#, sharey=True)
sns.barplot(ax=axs[0], data = sums.loc[order_list].reset_index(), x=0, y='industry', hue='level_2')
sns.barplot(ax=axs[1], data = sums_im.loc[order_im_list].reset_index(), x=0, y='industry', hue='level_2')
for j in range(2):
    axs[j].set_title(['Total', 'Imports'][j])
    axs[j].set_ylabel('')
    axs[j].set_xlabel('tCO2e')
    axs[j].set_xscale('log')
    for i in range(n):
        axs[j].axhline(0.5+i, c='k', linestyle=':')
        
fig.tight_layout()
plt.savefig(plot_filepath + 'barplot_CO2_global_by_sector_GHG.png', dpi=200, bbox_inches='tight')
plt.show()

        
######################
## Pct distribution ##
######################
n = 10

# total
pct = summary.sum(axis=0, level=['industry', 'year']).unstack('year').apply(lambda x: x/x.sum() * 100).stack('year')

order = pd.DataFrame(pct.mean(axis=0, level='industry').mean(1)).sort_values(0, ascending=False)
order['cumsum'] = order[0].cumsum()
order['cumpct'] = order['cumsum'] / order[0].sum() * 100
order_list = order.iloc[:n].index.tolist()

pct = pd.DataFrame(pct.stack())

#imports
pct_im = summary_im.sum(axis=0, level=['industry', 'year']).unstack('year').apply(lambda x: x/x.sum() * 100).stack('year')
order_im = pd.DataFrame(pct_im.mean(axis=0, level='industry').mean(1)).sort_values(0, ascending=False)
order_im['cumsum'] = order_im[0].cumsum()
order_im['cumpct'] = order_im['cumsum'] / order_im[0].sum() * 100
order_im_list = order_im.iloc[:n].index.tolist()

pct_im = pd.DataFrame(pct_im.stack())


#plot
fig, axs = plt.subplots(figsize=(17, 10), ncols=2)#, sharey=True)
sns.barplot(ax=axs[0], data = pct.loc[order_list].reset_index(), x=0, y='industry', hue='level_2')
sns.barplot(ax=axs[1], data = pct_im.loc[order_im_list].reset_index(), x=0, y='industry', hue='level_2')
for j in range(2):
    axs[j].set_title(['Total', 'Imports'][j])
    axs[j].set_ylabel('')
    axs[j].set_xlabel('Percentage of tCO2e')
    for i in range(n):
        axs[j].axhline(0.5+i, c='k', linestyle=':')

fig.tight_layout()
plt.savefig(plot_filepath + 'barplot_CO2_global_by_sector_GHG_pct.png', dpi=200, bbox_inches='tight')
plt.show()