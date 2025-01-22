
import pandas as pd
from sys import platform
import pickle
import matplotlib.pyplot as plt

# set working directory
# make different path depending on operating system
if platform[:3] == 'win':
    wd = 'O://'
else:
    wd = r'/Volumes/a72/' 
    
level =  'industry'

# define filepaths
data_filepath = wd + 'ESCoE_Project/data/'
emissions_filepath = wd + 'ESCoE_Project/data/Emissions/'
outputs_filepath = wd + 'ESCoE_Project/outputs/compare_all_outputs/'

# Dictonaries
country_dict = {'United Kingdom':'UK', 'Czech Republic':'Czechia', 'United States':'USA', 'Rest of the World':'RoW'}
data_dict = {'oecd':'ICIO', 'exio':'Exiobase', 'figaro':'Figaro','gloria':'Gloria'}

##################
## Run Analysis ##
##################
# load data
# direct
emissions_filepath = wd + 'ESCoE_Project/data/Emissions/'
ghg_direct = pickle.load(open(emissions_filepath + 'uk_direct.p', 'rb'))

# indirect
# FIGARO
ghg_indirect = pickle.load(open(emissions_filepath + 'Figaro/Figaro_industry_ghg_v2024_uk.p', 'rb'))
# pickle.dump(co2_gloria_prod, open(emissions_filepath + 'Gloria/Gloria_products_' + footprint + '_v' + version + '_uk.p', 'wb'))

ghg_figaro = pd.DataFrame(index=['Domestic', 'Imports'])
for year in list(ghg_indirect.keys()):
    temp = ghg_indirect[year].T['GB'].sum(axis=1).sum(axis=0, level=0).reset_index()
    temp['Origin'] = 'Imports'
    temp.loc[temp['index'] == 'GB', 'Origin']  = 'Domestic'
    temp = temp.groupby(['Origin']).sum()
    temp.loc['Domestic', 0] += ghg_direct[year].sum()
    ghg_figaro[year] = temp[0]

ghg_figaro = ghg_figaro.T

ghg_figaro.to_csv(emissions_filepath + 'Figaro_total_uk.csv')

ghg_figaro['total'] = ghg_figaro.sum(1)

# combine
ghg_figaro.plot(stacked=True); plt.show()


# GLORIA
ghg_indirect = pickle.load(open(emissions_filepath + 'Gloria/Gloria_industry_ghg_v2024_uk.p', 'rb'))
# pickle.dump(co2_gloria_prod, open(emissions_filepath + 'Gloria/Gloria_products_' + footprint + '_v' + version + '_uk.p', 'wb'))

ghg_gloria = pd.DataFrame(index=['Domestic', 'Imports'])
for year in list(ghg_indirect.keys()):
    temp = ghg_indirect[year]['United Kingdom'].sum(axis=1).sum(axis=0, level=0).reset_index()
    temp['Origin'] = 'Imports'
    temp.loc[temp['index'] == 'United Kingdom', 'Origin']  = 'Domestic'
    temp = temp.groupby(['Origin']).sum()
    temp.loc['Domestic', 0] += ghg_direct[year].sum()
    ghg_gloria[year] = temp[0]

ghg_gloria = ghg_gloria.T

ghg_gloria.to_csv(emissions_filepath + 'Gloria_total_uk.csv')

ghg_gloria['total'] = ghg_gloria.sum(1)

# combine
ghg_gloria.plot(stacked=True); plt.show()


ghg_all = ghg_gloria.join(ghg_figaro,  lsuffix='_Gloria', rsuffix='_Figaro')

ghg_all[['Domestic_Gloria', 'Imports_Gloria', 'Domestic_Figaro', 'Imports_Figaro']].plot()

ghg_all[['total_Gloria', 'total_Figaro']].plot()


