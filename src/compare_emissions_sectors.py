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

data_comb = ['ICIO, Figaro', 'Exiobase, ICIO', 'ICIO, Gloria', 'Exiobase, Figaro', 'Figaro, Gloria', 'Exiobase, Gloria']

def calc_rmspe(x1, x2):
    pct_diff = ((x1/x2) - 1) * 100
    pct_sq = pct_diff **2
    mean_sq = np.mean(pct_sq)
    error = np.sqrt(mean_sq)
    return(error)


country_dict = {'United Kingdom':'UK', 'Czech Republic':'Czechia', 'United States':'USA', 'Rest of the World':'RoW'}
data_dict = {'oecd':'ICIO', 'exio':'Exiobase', 'gloria':'Gloria', 'figaro':'Figaro'}

sector_dict = {
    'Electricity, gas, steam and air conditioning supply' : 'Elect., gas, steam,\nair conditioning',
    'Other non-metallic mineral products' : 'Non-metallic\nmineral products',
    'Land transport and transport via pipelines' : 'Land transport',
    'Food products, beverages and tobacco' : 'Food, beverages\nand tobacco',
    'Chemical, parmaceuticals and botanical products' : 'Chemicals,\nparmaceuticals,\nbotanicals',
    'Coke and refined petroleum products' : 'Coke, refined\npetroleum',
    'Machinery, computer, electronic, optical equipment, and other machinery and equipment' : 'Machinery and\nequipment',
    'Public administration and defence; compulsory social security' : 'Compulsory social\nsecurity',
    'Mining and quarrying, energy producing products' : 'Mining and\nquarrying',
    'Agriculture, hunting, forestry' : 'Agriculture,\nhunting, forestry',
    'Manufacturing nec; repair and installation of machinery and equipment' : 'Manufacturing nec',
    'Human health and social work activities' : 'Human care,\nsocial work',
    'Wholesale and retail trade; repair of motor vehicles' : 'Wholesale,\nretail trade',
    'Motor vehicles, trailers and semi-trailers' : 'Motor vehicles,\ntrailers',
    'Administrative, support and other professional and supporting transport services' : 'Professional\nservices',
    'Textiles, textile products, leather and footwear' : 'Textiles,\nleather products',
    'Other service activities' : 'Other services',
    'IT, information, postal, communication services and publishing' : 'IT, communication,\npublishing services',
    'Accommodation and food service activities' : 'Accomm., food\nservicess',
    'Rubber and plastics products' : 'Rubber, plastics\nproducts',
    'Paper products and printing' : 'Paper products',
    'Water supply; sewerage, waste management and remediation activities' : 'Water supply,\nsewerage, waste',
    'Fabricated metal products' : 'Metal products',
    'Other transport equipment' : 'Other transport\nequipment',
    'Financial and insurance activities' : 'Finance, insurance\nactivities',
    'Fishing and aquaculture' : 'Fishing, aquaculture',
    'Wood and products of wood and cork' : 'Wood products',
    'Activities of households as employers; undifferentiated goods- and services-producing activities of households for own use':'Households as\nemployers'
     }


# get openness of economy for ordering graphs
openness = pd.read_excel(data_filepath + 'lookups/lookup_trade_openness.xlsx', sheet_name='agg_data')
openness = openness.loc[openness['Countries'] != 'ROW Mean'].sort_values('Trade_openness_2018', ascending=False)

country_order = openness['combined_name'].tolist()

openness['country'] = country_order
           
###############
## Summarise ##
###############

# Total

summary = pd.DataFrame()
for year in years:
    temp_oecd = pd.DataFrame(co2_all['oecd'][year].sum(axis=1, level=0).sum(axis=0, level=1).unstack()).rename(columns={0:'oecd'})
    temp_figaro = pd.DataFrame(co2_all['figaro'][year].sum(axis=1, level=0).sum(axis=0, level=1).unstack()).rename(columns={0:'figaro'})
    temp_gloria = pd.DataFrame(co2_all['gloria'][year].sum(axis=1, level=0).sum(axis=0, level=1).unstack()).rename(columns={0:'gloria'})
    temp_exio = pd.DataFrame(co2_all['exio'][year].sum(axis=1, level=0).sum(axis=0, level=1).unstack()).rename(columns={0:'exio'})
    # merge all
    temp = temp_oecd.join(temp_figaro).join(temp_gloria).join(temp_exio) 
    temp['year'] = year
    summary = summary.append(temp.reset_index())
summary = summary.rename(columns={'level_0':'country', 'level_1':'sector'}).set_index(['country', 'year', 'sector']).rename(index=country_dict).rename(columns=data_dict)
    
# Imports

summary_im = pd.DataFrame()
for year in years:
    temp = {}
    for item in ['oecd', 'figaro', 'gloria', 'exio']:
        temp[item] = co2_all[item][year]
        for country in temp[item].index.levels[0]:
            temp[item].loc[country, country] = 0
        temp[item] = pd.DataFrame(temp[item].sum(axis=1, level=0).sum(axis=0, level=1).unstack()).rename(columns={0:item})
    # merge all
    temp_all = temp['oecd'].join(temp['figaro']).join(temp['gloria']).join(temp['exio'])
    temp_all['year'] = year
    summary_im = summary_im.append(temp_all.reset_index())
summary_im = summary_im.rename(columns={'level_0':'country', 'level_1':'sector'}).set_index(['country', 'year', 'sector']).rename(index=country_dict).rename(columns=data_dict)

summary_all = {'Total':summary, 'Imports':summary_im}

# Get means

mean_co2 = {'Total' : pd.DataFrame(summary.mean(axis=0, level=['sector', 'country']).mean(axis=1)).rename(columns={0:'mean_co2'}), 
            'Imports' : pd.DataFrame(summary_im.mean(axis=0, level=['sector', 'country']).mean(axis=1)).rename(columns={0:'mean_co2'})}


mean_co2_sector = {}
for data in ['Total', 'Imports']:
    mean_co2_sector[data] = mean_co2[data].sum(axis=0, level='sector').sort_values('mean_co2', ascending=False)
    mean_co2_sector[data]['pct_co2'] = mean_co2_sector[data]['mean_co2'] / mean_co2_sector[data]['mean_co2'].sum() * 100
    mean_co2_sector[data]['pct_co2_cumu'] = mean_co2_sector[data]['pct_co2'].cumsum()


temp = mean_co2_sector['Total'].join(mean_co2_sector['Imports'], rsuffix='_import')

##############
# Start Loop #
##############

sectors = {x:mean_co2_sector[x].index.tolist() for x in ['Total', 'Imports']}

results = pd.DataFrame()
top_sectors = {}; data_rmspe = {}; data_direction = {}

for data in ['Total', 'Imports']:
    top_sectors[data] = sectors[data][:10]
    for sector in sectors[data]:
        
        #############################
        ## Change in trend - RMSPE ##
        #############################
        
        # Total
        temp = summary_all[data].unstack('country').swaplevel(axis=1).swaplevel(axis=0).loc[sector]
        data_rmspe_temp = pd.DataFrame(columns=['country'])
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
                data_rmspe_temp = data_rmspe_temp.append(temp3)
        data_rmspe_temp = data_rmspe_temp.merge(data_rmspe_temp.groupby('country').mean().reset_index().rename(columns={'RMSPE':'mean'}), on='country').sort_values(['mean', 'dataset'])
       
        data_rmspe[data] = data_rmspe_temp
        
        #################################
        ## Change in trend - Direction ##
        #################################
        
        # Total
        temp = summary.fillna(0).unstack('year').swaplevel(axis=1).swaplevel(axis=0).loc[sector].stack(level=1)
        data_direction_temp = temp[years[1:]]
        for year in years[1:]:
            data_direction_temp[year] = temp[year] / temp[year - 1]
        data_direction_temp = data_direction_temp.unstack(level=1).stack(level=0)
        # Convert to True vs False
        for comb in data_comb:
            d0 = comb.split(', ')[0]
            d1 = comb.split(', ')[1]
            data_direction_temp[comb] = False
            data_direction_temp.loc[((data_direction_temp[d0]>1) & (data_direction_temp[d1]>1) | (data_direction_temp[d0]<1) & (data_direction_temp[d1]<1) | 
                                (data_direction_temp[d0]==1) & (data_direction_temp[d1]==1)), comb] = True
        data_direction_temp = data_direction_temp[data_comb].stack().reset_index().rename(columns={'level_2':'dataset', 0:'Same_direction'})
        data_direction_temp['count'] = 1
        data_direction_temp = data_direction_temp.set_index(['country', 'year', 'dataset', 'Same_direction']).unstack('Same_direction')\
            .droplevel(axis=1, level=0).fillna(0).sum(axis=0, level=[0, 2])
        data_direction_temp['pct_same'] = data_direction_temp[True] / (data_direction_temp[True] + data_direction_temp[False])*100
        
        # Combine all
        data_direction[data] = data_direction_temp
        
        
        # save as results
        mean = mean_co2[data].loc[sector]
        
        plot_data = data_direction[data].reset_index().merge(data_rmspe[data], on =['country', 'dataset'])
        plot_data = plot_data[['country', 'dataset', 'pct_same', 'RMSPE']].merge(mean, on='country')
        plot_data['Country'] = '                     ' + plot_data['country']
        
        plot_data['Type'] = data
        plot_data['Sector'] = sector
        results = results.append(plot_data.reset_index())


###################
## Plot together ##
###################

fs = 16
pal = 'tab10'
c_box = '#000000'
c_vlines = '#B9B9B9'
point_size = 9

# Stripplots
for sector in top_sectors['Total'][:3]:      
    fig, axs = plt.subplots(nrows=2, figsize=(20, 10), sharex=True)
    order = results.loc[(results['Type'] == 'Total') & (results['Sector'] == sector)].sort_values('mean_co2', ascending=False)['country'].unique()
    for i in range(2):
        data = ['Total', 'Imports'][i]
        plot_data = results.loc[(results['Type'] == data) & (results['Sector'] == sector)].sort_values('dataset').set_index('country').loc[order].reset_index()
       
        sns.stripplot(ax=axs[i], data=plot_data, x='Country', y='RMSPE', hue='dataset', s=point_size, jitter=0.4, palette=pal); 
        axs[i].set_xlabel('')
        axs[i].tick_params(axis='y', labelsize=fs)
        axs[i].set_yscale('log')
        axs[i].set_ylim(0, 10**5)
        
        if data == 'Total':
            ax_twin0 = axs[i].twinx()
            sns.lineplot(ax=ax_twin0, data=plot_data[['country', 'mean_co2']].drop_duplicates(), y='mean_co2', x='country', color='k')
            ax_twin0.tick_params(axis='y', labelsize=fs)
            ax_twin0.set_ylabel('Total emissions (CO2)', fontsize=fs); 
        else:
            ax_twin1 = axs[i].twinx()
            sns.lineplot(ax=ax_twin1, data=plot_data[['country', 'mean_co2']].drop_duplicates(), y='mean_co2', x='country', color='k')
            ax_twin1.tick_params(axis='y', labelsize=fs)
            ax_twin1.set_ylabel('Imported emissions (CO2)', fontsize=fs); 
        
    axs[0].set_ylabel('Total emissions RMSPE (%)', fontsize=fs)
    axs[1].set_ylabel('Imported emissions RMSPE (%)', fontsize=fs)
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), fontsize=fs, ncol=len(plot_data['dataset'].unique()))
    axs[1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), fontsize=fs, ncol=len(plot_data['dataset'].unique()))
    
    axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=90, va='center', fontsize=fs); 
    axs[1].xaxis.set_ticks_position('top') # the rest is the same

    for c in range(len(plot_data['country'].unique())-1):
        axs[0].axvline(c+0.5, c=c_vlines, linestyle=':')
        axs[1].axvline(c+0.5, c=c_vlines, linestyle=':')
    
    fig.tight_layout()
    plt.savefig(plot_filepath + 'Stripplot_similarity_bycountry_' + sector + '.png', dpi=200, bbox_inches='tight')
    plt.show()


# plot with sector on the x 

# Boxplot

for data in ['Total', 'Imports']:
    
    top_sectors = sectors[data][:10]
    
    fig, axs = plt.subplots(nrows=2, figsize=(20, 10), sharex=True)
        
    temp = results.loc[results['Type'] == data].set_index('Sector').loc[top_sectors]
    temp2 = temp.groupby(['Sector', 'country']).mean().sum(axis=0, level='Sector')[['mean_co2']]\
        .loc[top_sectors].rename(index=sector_dict).reset_index()
    
    temp = temp.rename(index=sector_dict).reset_index()
    temp['Sector'] = temp['Sector'] + '\n\n'
    
    # Boxplot
    # RMSPE
    sns.boxplot(ax=axs[0], data=temp, hue='dataset', y='RMSPE', x='Sector', showfliers=True)
    axs[0].set_xlabel('')
    axs[0].tick_params(axis='y', labelsize=fs)
    axs[0].set_yscale('log')
    axs[0].set_ylim(0, 10**5)
    
    ax_twin = axs[0].twinx()
    sns.lineplot(ax=ax_twin, data=temp2, y='mean_co2', x='Sector', color='k')
    ax_twin.tick_params(axis='y', labelsize=fs)
    
    if data == 'Total':
        ax_twin.set_ylabel('Total emissions (CO2)', fontsize=fs); 
    else:
        ax_twin.set_ylabel('Imported emissions (CO2)', fontsize=fs); 
        
    # Direction 
    temp = results.loc[results['Type'] == data].set_index('Sector').loc[top_sectors]
    temp2 = temp.groupby(['Sector', 'country']).mean().sum(axis=0, level='Sector')[['mean_co2']]\
        .loc[top_sectors].rename(index=sector_dict).reset_index()
    
    temp = temp.rename(index=sector_dict).reset_index()
    temp['Sector'] = temp['Sector'] + '\n\n'
    
    
    sns.boxplot(ax=axs[1], data=temp, hue='dataset', y='pct_same', x='Sector', showfliers=True)
    axs[1].set_xlabel('')
    axs[1].tick_params(axis='y', labelsize=fs)
    axs[1].set_ylim(-5, 105)
    
    ax_twin = axs[1].twinx()
    sns.lineplot(ax=ax_twin, data=temp2, y='mean_co2', x='Sector', color='k')
    ax_twin.tick_params(axis='y', labelsize=fs)
    
    if data == 'Total':
        ax_twin.set_ylabel('Total emissions (CO2)', fontsize=fs); 
    else:
        ax_twin.set_ylabel('Imported emissions (CO2)', fontsize=fs); 
        
    # Labels
    
    axs[0].set_ylabel('RMSPE (%)', fontsize=fs)
    axs[1].set_ylabel('Similarity direction (%)', fontsize=fs)
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), fontsize=fs, ncol=6)
    axs[1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), fontsize=fs, ncol=6)
    
    axs[1].set_xticklabels(axs[1].get_xticklabels(), va='center', fontsize=fs); 
    axs[1].xaxis.set_ticks_position('top') # the rest is the same
    
    for c in range(len(temp['Sector'].unique())):
        axs[0].axvline(c+0.5, c=c_vlines, linestyle=':')
        axs[1].axvline(c+0.5, c=c_vlines, linestyle=':')
    fig.tight_layout()
    plt.savefig(plot_filepath + 'Boxplot_bysector_' + data + '.png', dpi=200, bbox_inches='tight')
    plt.show()

####

summary = cp.copy(results)
summary.replace([np.inf, -np.inf], 0, inplace=True)
summary = summary.groupby(['Type', 'Sector', 'dataset']).describe()[
    [('pct_same', 'mean'), ('pct_same', 'std'), ('RMSPE', 'mean'), 
     ('RMSPE', 'std'), ('mean_co2', 'mean'), ('mean_co2', 'std')]]
summary = summary.unstack(level=2).drop([
    ('mean_co2', 'mean', 'Exiobase, Gloria'), ('mean_co2', 'mean', 'Exiobase, ICIO'),
    ('mean_co2', 'mean', 'Figaro, Gloria'), ('mean_co2', 'mean', 'ICIO, Figaro'),
    ('mean_co2', 'mean', 'ICIO, Gloria'), ('mean_co2', 'std', 'Exiobase, Gloria'),
    ('mean_co2', 'std', 'Exiobase, ICIO'), ('mean_co2', 'std', 'Figaro, Gloria'),
    ('mean_co2', 'std', 'ICIO, Figaro'), ('mean_co2', 'std', 'ICIO, Gloria')], axis=1)



aaaaaaaab = summary.unstack('sector').swaplevel(axis=1)[['Electricity, gas, steam and air conditioning supply']]




