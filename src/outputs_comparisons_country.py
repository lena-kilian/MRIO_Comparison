# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 13:13:18 2024

@author: geolki
"""

import pandas as pd
from sys import platform
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import get_cmap

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

order_var = 'prop_imports' # 'gdp' # 'openness' # 'prop_imports' # 'ghg_cap_total' # 'ghg_cap_imports'
same_direction_pct_cutoff = 1.5 

agg_vars = ['agg_after']#, 'agg_before']
levels = ['industry', 'products']

for level in levels:
    for agg_var in agg_vars:   
        # import data
        summary_co2 = pickle.load(open(outputs_filepath + 'summary_co2_country_' + level + '_' + agg_var + '.p', 'rb'))
        mean_co2 = pickle.load(open(outputs_filepath + 'mean_co2_country_' + level + '_' + agg_var + '.p', 'rb'))
        rmse_pct = pickle.load(open(outputs_filepath + 'rmse_pct_country_' + level + '_' + agg_var + '.p', 'rb'))
        direction = pickle.load(open(outputs_filepath + 'direction_annual_country_' + level + '_' + agg_var + '.p', 'rb'))
        reg_results = pickle.load(open(outputs_filepath + 'regression_country_' + level + '_' + agg_var + '.p', 'rb'))
        
        reg_validation = pickle.load(open(outputs_filepath + 'regression_validation_' + level + '_' + agg_var + '.p', 'rb'))
        reg_fit = {item:pd.read_csv(outputs_filepath + 'regression_country_linearity_AIC_' + item + '_' + level + '.csv') for item in ['Total', 'Imports']}
        
        country_order = pickle.load(open(outputs_filepath + 'country_order_' + level + '_' + agg_var + '.p', 'rb'))[order_var]
        datasets = summary_co2['Total'].columns.tolist(); datasets.sort()
        years = summary_co2['Total'].index.levels[0].tolist()
        
        data_comb = []
        for i in range(len(datasets)):
            for j in range(i+1, len(datasets)):
                data_comb.append(datasets[i] + ', ' + datasets[j])
        
        # plot params
        fs = 16
        pal = 'tab10'
        c_box = '#000000'
        c_vlines = '#6d6d6d'
        point_size = 20
        scatter_size = 100
        pal = 'tab10'
        marker_list = ["o", "X", "s", "P"]
        
        ##########################
        ## Total & Pct imported ##
        ##########################
        
        # global imports
        percent_im_global = summary_co2['Imports'].sum(axis=0, level=1).mean(axis=0) / summary_co2['Total'].sum(axis=0, level=1).mean(axis=0) * 100
        
        # country imports
        percent_im = pd.DataFrame((summary_co2['Imports'].mean(axis=0, level=0) / summary_co2['Total'].mean(axis=0, level=0) * 100)\
                                  .stack()).reset_index().rename(columns={'level_1':'dataset', 0:'pct_im'})
        plot_data = mean_co2['Total'].stack().reset_index().rename(columns={'level_1':'dataset', 0:'mean_co2'})\
            .merge(percent_im, on=['country', 'dataset'])
        plot_data['country_cat'] = pd.Categorical(plot_data['country'], categories=country_order, ordered=True)
        plot_data['dataset_cat'] = pd.Categorical(plot_data['dataset'], categories=datasets, ordered=True)
            
        plot_data['Country'] = '                     ' + plot_data['country']
        
        
        plot_data = plot_data.sort_values(['country_cat', 'dataset_cat'])
        
        # Scatterplot
        fig, axs = plt.subplots(nrows=2, figsize=(15, 7.5), sharex=True)
        
        sns.scatterplot(ax=axs[0], data=plot_data, x='country', y='mean_co2', hue='dataset', s=scatter_size, palette=pal, style='dataset', markers=marker_list)
        axs[0].set_ylabel('Footprint (ktCO\N{SUBSCRIPT TWO}e)', fontsize=fs); 
        axs[0].set_yscale('log')
        
        sns.scatterplot(ax=axs[1], data=plot_data, x='country', y='pct_im', hue='dataset', s=scatter_size, palette=pal, style='dataset', markers=marker_list)
        axs[1].tick_params(axis='y', labelsize=fs)
        axs[1].set_ylabel('Emisions imported (%)', fontsize=fs); 
        
        #axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), fontsize=fs, ncol=len(datasets), markerscale=2)
        axs[0].legend().remove()
        axs[1].legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), fontsize=fs, ncol=len(datasets), markerscale=2)
        
        for i in range(2):
            axs[i].tick_params(axis='y', labelsize=fs)
            axs[i].set_xlabel('', fontsize=fs)
            for c in range(len(plot_data['country'].unique())-1):
                axs[i].axvline(c+0.5, c=c_vlines, linestyle=':')
            
        axs[1].set_xticklabels(plot_data['Country'].unique(), rotation=90, va='center', fontsize=fs); 
        axs[1].xaxis.set_ticks_position('top') # the rest is the same
        
        fig.tight_layout()
        plt.savefig(plot_filepath + 'scatterplot_overview_bycountry_GHG_' + order_var + '_' + level + '_' + agg_var + '.png', dpi=200, bbox_inches='tight')
        plt.show()
        
        
        ###################################
        ## Change in trend - RMSE / Mean ##
        ###################################
        
        plot_data = rmse_pct['Total'].set_index('country').stack().reset_index().rename(columns={0:'Value'})
        plot_data['Type'] = 'Total'
        temp = rmse_pct['Imports'].set_index('country').stack().reset_index().rename(columns={0:'Value'})
        temp['Type'] = 'Imports'
        
        plot_data = plot_data.append(temp)
        
        # Histogram
        
        fig, axs = plt.subplots(nrows=len(data_comb), ncols=2, figsize=(8, 10), sharey=True, sharex=True)
        for c in range(2):
            item = ['Total', 'Imports'][c]
            for r in range(len(data_comb)):
                temp = plot_data.loc[(plot_data['Type'] == item) & (plot_data['dataset'] == data_comb[r])]
                sns.histplot(ax=axs[r, c], data=temp, x='Value', binwidth=10, color=get_cmap(pal)(c), alpha=0.5)
                axs[r, c].set_ylabel(data_comb[r].replace(', ', ',\n'), fontsize=fs)
                axs[r, c].axvline(temp.median().values, c='k', linestyle=':', linewidth=2)
                #y_labels =[int(y) for y in axs[r, c].get_yticks()]
                #axs[r, 0].set_yticklabels(y_labels, fontsize=fs); 
                axs[r, 0].tick_params(axis='y', labelsize=fs)
            #x_labels = [int(x) for x in axs[r, c].get_xticks()]
            #axs[r, c].set_xticklabels(x_labels, fontsize=fs); 
            axs[r, c].tick_params(axis='x', labelsize=fs)
            axs[0, c].set_title(item, fontsize=fs)
            axs[r, c].set_xlabel("NRMSE (%)\n", fontsize=fs)
        fig.tight_layout()
        plt.savefig(plot_filepath + 'histplot_similarity_bydata_rmse_pct_GHG_' + level + '_' + agg_var + '.png', dpi=200, bbox_inches='tight')
        plt.show() 
        
        #################################
        ## Change in trend - Direction ##
        #################################
        
        plot_data = direction['Total']; plot_data['Type'] = 'Total'
        temp = direction['Imports']; temp['Type'] = 'Imports'
        
        plot_data = plot_data.append(temp)
        
        # Histogram
        
        fig, axs = plt.subplots(nrows=len(data_comb), ncols=2, figsize=(8, 10), sharex=True, sharey=True)
        for c in range(2):
            item = ['Total', 'Imports'][c]
            for r in range(len(data_comb)):
                temp = plot_data.loc[(plot_data['Type'] == item) & (plot_data['dataset'] == data_comb[r])]
                sns.histplot(ax=axs[r, c], data=temp, x='pct_same', binwidth=10, color=get_cmap(pal)(c), alpha=0.5)
                axs[r, c].set_ylabel(data_comb[r].replace(', ', ',\n'), fontsize=fs)
                axs[r, c].axvline(temp.median()['pct_same'], c='k', linestyle=':', linewidth=2)
                #y_labels =[int(y) for y in axs[r, c].get_yticks()]
                #axs[r, 0].set_yticklabels(y_labels, fontsize=fs);
                axs[r, 0].tick_params(axis='y', labelsize=fs)
            #x_labels = [int(x) for x in axs[r, c].get_xticks()]
            #axs[r, c].set_xticklabels(x_labels, fontsize=fs); 
            axs[r, c].tick_params(axis='x', labelsize=fs)
            axs[0, c].set_title(item, fontsize=fs)
            axs[r, c].set_xlabel("Annual Direction\nSimilarity (%)", fontsize=fs)
        fig.tight_layout()
        plt.savefig(plot_filepath + 'histplot_similarity_bydata_Boxplot_similarity_bydata_direction_GHG_' + level + '_' + agg_var + '.png', dpi=200, bbox_inches='tight')
        plt.show() 
        
        
        ###################################
        ## Regress footprints over years ##
        ###################################
        
        fig, axs = plt.subplots(nrows=2, figsize=(15, 8), sharex=True)
        for r in range(2):
            item = ['Total', 'Imports'][r]
            plot_data = reg_results[item].drop('reg_validation_pct', axis=1)
            plot_data.loc[(plot_data['max'] <= same_direction_pct_cutoff) & (plot_data['min'] >= same_direction_pct_cutoff * -1), 'Same direction'] = True
            plot_data = plot_data.loc[country_order].set_index('Same direction', append=True).drop(['max', 'min'], axis=1)\
                .stack().reset_index().rename(columns={0:'Average pct change', 'level_2':'Data'})
            plot_data['Same direction:'] = pd.Categorical(plot_data['Same direction'], categories=[True, False], ordered=True)
            plot_data['Country'] = '                    ' + plot_data['country']
            plot_data['Data:'] = plot_data['Data']
            
            sns.scatterplot(ax=axs[r], data=plot_data, x='country', y='Average pct change', style='Data:', hue='Same direction:', s=scatter_size, palette=pal, markers=marker_list)
        
        axs[0].legend().remove()
        axs[1].legend(loc='lower center', columnspacing=0.5, bbox_to_anchor=(0.5, -0.25), fontsize=fs, ncol=8, markerscale=2)
        
        for i in range(2):
            axs[i].tick_params(axis='y', labelsize=fs)
            axs[i].set_xlabel('', fontsize=fs)
            axs[i].set_ylabel('Average yearly\nchange (%)', fontsize=fs); 
            axs[i].axhline(same_direction_pct_cutoff,  c='k', linestyle='--'); 
            axs[i].axhline(same_direction_pct_cutoff *-1, c='k', linestyle='--'); 
            axs[i].axhline(0, c='k');
            for c in range(len(plot_data['country'].unique())-1):
                axs[i].axvline(c+0.5, c=c_vlines, linestyle=':')
            
        axs[1].set_xticklabels(plot_data['Country'].unique(), rotation=90, va='center', fontsize=fs); 
        axs[1].xaxis.set_ticks_position('top') # the rest is the same
        
        fig.tight_layout()
        plt.savefig(plot_filepath + 'scatterplot_overview_regresults_GHG_' + order_var + '_' + level + '_' + agg_var + '.png', dpi=200, bbox_inches='tight')
        plt.show()
        
        
        
        #############################
        ## Longitudinal footprints ##
        #############################
        
        '''
        fig, axs = plt.subplots(nrows=len(country_order), ncols=2, figsize=(10, 120))
        for c in range(2):
            item = ['Total', 'Imports'][c]
            temp = summary_co2[item]
            
            for r in range(len(country_order)):
                country = country_order[r]
                plot_data = temp.loc[country].stack().reset_index().rename(columns={'level_1':'Datasets', 0:'tCO2'})
                sns.lineplot(ax=axs[r, c], data=plot_data, x='year', y='tCO2', hue='Datasets', legend=False)
                axs[r, c].set_title(country + ' - ' + item, fontsize=fs)
        plt.savefig(plot_filepath + 'Lineplot_CO2_all_' + order_var + '_' + agg_var + '.png', dpi=200, bbox_inches='tight')
        plt.show()
        '''
        
        
        ####################
        ## Regression Fit ##
        ####################
        
        reg_aic = pd.DataFrame()
        for item in ['Total', 'Imports']:
            for ds in datasets:
                temp = reg_fit[item].loc[reg_fit[item]['ds'] == ds]
                temp = temp.rename(columns={'aic1':'Linear', 'aic2':'Quadratic', 'aic3':'Cubic'})\
                    .set_index(['country', 'ds'])[['Linear', 'Quadratic', 'Cubic']].stack().reset_index()\
                        .rename(columns={'level_2':'Model fit', 0:'AIC', 'ds':'Data'})
                temp['Type'] = item
                reg_aic = reg_aic.append(temp)  
        reg_aic = reg_aic.set_index(['country', 'Data', 'Model fit', 'Type']).unstack(['Data', 'Model fit'])
        
        ##################################
        ## Regression Fit Years removed ##
        ##################################
        
        reg_val = pd.DataFrame()
        for item in ['Total', 'Imports']:
            temp = reg_validation[item].loc[reg_validation[item]['year'] != 0]
            temp = temp.groupby(['ds', 'country'])['coef'].describe()[['mean', 'std']]
            temp = temp.join(reg_validation[item].loc[reg_validation[item]['year'] == 0].set_index(['ds', 'country'])[['coef']])
            temp = temp[['coef', 'mean']].unstack('ds').swaplevel(axis=1) #  'std'
            temp['Type'] = item
            reg_val = reg_val.append(temp.set_index('Type', append=True))
        reg_val = reg_val.reindex(sorted(reg_val.columns), axis=1)