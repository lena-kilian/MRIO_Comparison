# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 10:30:46 2023

@author: geolki
"""

import pandas as pd
from sys import platform
import pickle

# set working directory
# make different path depending on operating system
if platform[:3] == 'win':
    wd = 'O://'
else:
    wd = r'/Volumes/a72/' 
  
# define params
corr_method = 'spearman' # 'pearson

agg_vars = ['agg_after']#, 'agg_before']
levels = ['industry', 'products']

# define filepaths
data_filepath = wd + 'ESCoE_Project/data/'
emissions_filepath = wd + 'ESCoE_Project/data/Emissions/'
outputs_filepath = wd + 'ESCoE_Project/outputs/compare_all_outputs/'

# Dictonaries
country_dict = {'United Kingdom':'UK', 'Czech Republic':'Czechia', 'United States':'USA', 'Rest of the World':'RoW'}
data_dict = {'oecd':'ICIO', 'exio':'Exiobase', 'figaro':'Figaro', 'gloria':'Gloria'}

ind_dict = {
    'Accommodation and food service activities' : 'Accommodation &\nfood services',
    'Activities of extraterritorial organisations and bodies' : 'Extraterritorial\norganisations',
    'Activities of households as employers; undifferentiated goods- and services-producing activities of households for own use' : 'Households as\nemployers',
    'Administrative, support and other professional and supporting transport services' : 'Professional &\nsupporting transport\nservices',
    'Agriculture, hunting, forestry' : 'Agriculture &\nforestry',
    'Air transport' : 'Air transport',
    'Basic metals' : 'Basic metals',
    'Chemical, parmaceuticals and botanical products' : 'Chemicals &\nparmaceuticals',
    'Coke and refined petroleum products' : 'Coke & refined\npetroleum',
    'Construction' : 'Construction',
    'Education' : 'Education',
    'Electricity, gas, steam and air conditioning supply' : 'Electricity &\ngas',
    'Fabricated metal products' : 'Fabricated metals',
    'Financial and insurance activities' : 'Finance &\ninsurance',
    'Fishing and aquaculture' : 'Fishing &\naquaculture',
    'Food products, beverages and tobacco' : 'Processing of\nfood & beverages',
    'Human health and social work activities' : 'Human health &\nsocial work',
    'IT, information, postal, communication services and publishing' : 'IT, communication\nservices & publishing',
    'Land transport and transport via pipelines' : 'Land & pipeline\ntransport',
    'Machinery, computer, electronic, optical equipment, and other machinery and equipment' : 'Machinery &\nequipment',
    'Manufacturing nec; repair and installation of machinery and equipment' : 'Repair &\nmanufacturing nec',
    'Mining and quarrying, energy producing products' : 'Mining',
    'Motor vehicles, trailers and semi-trailers' : 'Motor vehicles',
    'Other non-metallic mineral products' : 'Other non-\nmetallic minerals',
    'Other service activities' : 'Other services',
    'Other transport equipment' : 'Other transport\nequipment',
    'Paper products and printing' : 'Paper &\nprinting',
    'Public administration and defence; compulsory social security' : 'Public administra-\ntion & defence',
    'Real estate activities' : 'Real estate\nservices',
    'Rubber and plastics products' : 'Rubber &\nplastics',
    'Textiles, textile products, leather and footwear' : 'Textiles',
    'Water supply; sewerage, waste management and remediation activities' : 'Water supply &\nwaste management',
    'Water transport' : 'Water transport',
    'Wholesale and retail trade; repair of motor vehicles' : 'Wholesale & retail',
    'Wood and products of wood and cork' : 'Wood & cork'
            }

##################
## Run Analysis ##
##################

for level in levels:
    for agg_var in agg_vars:
        
        # load data
        co2_all = pickle.load(open(emissions_filepath + 'Emissions_' + level + '_all_' + agg_var + '.p', 'rb'))
            
        ######### REMOVE THIS LATER
        co2_all = {data:co2_all[data] for data in list(data_dict.keys())}
        ################
        
        # Variable lookups
        datasets = list(data_dict.values()); datasets.sort()
        years = list(co2_all[list(data_dict.keys())[0]].keys())
        
        data_comb = []
        for i in range(len(datasets)):
            for j in range(i+1, len(datasets)):
                data_comb.append(datasets[i] + ', ' + datasets[j])
        
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
            # rename index to match
            temp_oecd.index.names = ['industry', 'country']
            temp_figaro.index.names = ['industry', 'country']
            temp_exio.index.names = ['industry', 'country']
            temp_gloria.index.names = ['industry', 'country']
            # merge all
            temp = temp_oecd.join(temp_figaro, how='outer').join(temp_exio, how='outer').join(temp_gloria, how='outer').fillna(0)
            temp['year'] = year
            summary = summary.append(temp.reset_index())
        summary = summary.rename(columns={'level_0':'industry', 'level_1':'country'}).set_index(['country', 'industry'])\
            .rename(index=country_dict).swaplevel(axis=0).rename(index=ind_dict).set_index('year', append=True).rename(columns=data_dict)
            
        # Imports
        summary_im = pd.DataFrame()
        for year in years:
            temp = {}
            for item in list(co2_all.keys()):
                temp[item] = co2_all[item][year]
                for country in temp[item].index.levels[0]:
                    temp[item].loc[country, country] = 0
                temp[item] = pd.DataFrame(temp[item].sum(axis=1, level=0).sum(axis=0, level=1).stack()).rename(columns={0:item})
                # rename index to match
                temp[item].index.names = ['industry', 'country']
            
            # merge all
            temp_all = temp['oecd'].join(temp['figaro'], how='outer').join(temp['exio'], how='outer').join(temp['gloria'], how='outer').fillna(0)
            temp_all['year'] = year
            summary_im = summary_im.append(temp_all.reset_index())
        summary_im = summary_im.rename(columns={'level_0':'industry', 'level_1':'country'}).set_index(['country', 'industry'])\
            .rename(index=country_dict).swaplevel(axis=0).rename(index=ind_dict).set_index('year', append=True).rename(columns=data_dict)
        
        
        # Combine all
        summary_industry = {'Total':summary, 'Imports':summary_im}
        
        # save
        pickle.dump(summary_industry, open(outputs_filepath + 'summary_' + level + '_' + agg_var + '.p', 'wb'))
        
        ###################
        ## Industry Corr ##
        ###################
        
        # get mean emissions by sector and country
        # total
        means = summary.mean(axis=0, level=['country', 'industry'])
        corr = means.reset_index().groupby(['country']).corr(method=corr_method).unstack(level=1)
        corr.columns = [x[0] + ', ' + x[1] for x in corr.columns.tolist()]
        corr = corr[data_comb]
        
        corr = corr.stack().reset_index().rename(columns={'level_1':'Data', 0:corr_method})
        
        # imports
        means_im = summary_im.mean(axis=0, level=['country', 'industry'])
        corr_im = means_im.reset_index().groupby(['country']).corr(method=corr_method).unstack(level=1)
        corr_im.columns = [x[0] + ', ' + x[1] for x in corr_im.columns.tolist()]
        corr_im = corr_im[data_comb]
        
        corr_im = corr_im.stack().reset_index().rename(columns={'level_1':'Data', 0:corr_method})
            
        # Combine all
        corr_all = {'Total':corr, 'Imports':corr_im}
        
        # save
        pickle.dump(corr_all, open(outputs_filepath + 'corr_' + level + '_' + agg_var + '.p', 'wb'))
