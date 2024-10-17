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
    
corr_method = 'spearman' # 'pearson

# define filepaths
data_filepath = wd + 'ESCoE_Project/data/'
emissions_filepath = wd + 'ESCoE_Project/data/Emissions/'
outputs_filepath = wd + 'ESCoE_Project/outputs/compare_all_outputs/'

co2_all = pickle.load(open(emissions_filepath + 'Emissions_aggregated_all.p', 'rb'))

datasets = list(co2_all.keys())
years = list(co2_all[datasets[0]].keys())

# data lookups
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


# Combine all
summary_industry = {'Total':summary, 'Imports':summary_im}

# save
pickle.dump(summary_industry, open(outputs_filepath + 'summary_industry.p', 'wb'))

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
pickle.dump(corr_all, open(outputs_filepath + 'corr_industry.p', 'wb'))
