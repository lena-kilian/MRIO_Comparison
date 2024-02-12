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
import copy as cp
import numpy as np

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

data_comb_cols = [(x.split(', ')[0], x.split(', ')[1]) for x in data_comb]

# change countries and sectors to ukmrio compatible
lookup = pd.read_excel('O:/ESCoE_Project/data/lookups/combined_to_ukmrio_lookup.xlsx', sheet_name=None)

country_dict = dict(zip(lookup['countries']['combined'], lookup['countries']['ukmrio']))
sector_dict = dict(zip(lookup['sectors']['combined_name'], lookup['sectors']['ukmrio_name']))

###############
## Summarise ##
###############

country = 'United Kingdom'

total = pd.DataFrame()
for year in years:
    temp_oecd = pd.DataFrame(co2_all['oecd'][year][[country]].sum(axis=1, level=0)).rename(columns={country:'oecd'})
    temp_figaro = pd.DataFrame(co2_all['figaro'][year][[country]].sum(axis=1, level=0)).rename(columns={country:'figaro'})
    temp_gloria = pd.DataFrame(co2_all['gloria'][year][[country]].sum(axis=1, level=0)).rename(columns={country:'gloria'})
    temp_exio = pd.DataFrame(co2_all['exio'][year][[country]].sum(axis=1, level=0)).rename(columns={country:'exio'})
    
    temp = temp_oecd.join(temp_figaro).join(temp_gloria).join(temp_exio)
    
    # aggregate to ukmrio levels
    c_new = [country_dict[x[0]] for x in temp.index.tolist()]
    s_new = [sector_dict[x[1]] for x in temp.index.tolist()]
    temp.index = pd.MultiIndex.from_arrays([c_new, s_new])
    
    temp = temp.sum(axis=0, level=[0,1])
    
    temp['year'] = year
    
    total = total.append(temp.reset_index())
    
total = total.rename(columns={'level_0':'country', 'level_1':'sector'}).set_index(['country', 'sector', 'year']).fillna(0)


domestic = total.loc[country, :]
imports = total.drop(country)

for item in ['total', 'domestic', 'imports']:
    data = eval(item).reset_index().groupby(['year']).sum()[datasets]
    data.plot()
    plt.title(country + ' ' + item); plt.legend(bbox_to_anchor=(1,1))
    plt.ylabel('Carbon emissions')
    plt.savefig(plot_filepath + country + '_lineplot_country_emissions_' + item + '.png', dpi=200, bbox_inches='tight')

    data = data.T
    temp = cp.copy(data)
    for year in data.columns.tolist()[1:]:
        data[str(year-1) + '-' + str(year)] = (temp[year] - temp[year-1]) / temp[year-1] * 100
    data = data.drop(years, axis=1)
    data.T.plot(kind='bar')
    plt.axhline(0, linestyle='--', c='k')
    plt.title(country + ' ' + item); plt.legend(bbox_to_anchor=(1,1)); plt.ylabel('Percentage change')
    plt.savefig(plot_filepath + country + '_lineplot_country_emissions_change_' + item + '.png', dpi=200, bbox_inches='tight')

#################
## Correlation ##
#################

corr_total_ghg = total.groupby('year').sum().corr().unstack()[data_comb_cols]

corr_imports_ghg = imports.groupby('year').sum().corr().unstack()[data_comb_cols]

temp = total.mean(axis=0, level=['country', 'sector']).mean(axis=1)
corr_all = (total + 0.01).groupby(['country', 'sector']).corr().unstack(level=2)[data_comb_cols]
corr_all[('mean', 'mean')] = temp
corr_all = corr_all.sort_values(('mean', 'mean'), ascending=False)

temp = total.mean(axis=0, level=['country', 'sector']).sum(axis=0, level='country').mean(axis=1)
corr_country = total.sum(axis=0, level=['country', 'year']).groupby(['country']).corr().unstack(level=1)[data_comb_cols]
corr_country[('mean', 'mean')] = temp
corr_country = corr_country.sort_values(('mean', 'mean'), ascending=False)
  
temp = total.mean(axis=0, level=['country', 'sector']).sum(axis=0, level='sector').mean(axis=1)
corr_sector_total = total.sum(axis=0, level=['sector', 'year']).groupby(['sector']).corr().unstack(level=1)[data_comb_cols]
corr_sector_total[('mean', 'mean')] = temp 
corr_sector_total = corr_sector_total.sort_values(('mean', 'mean'), ascending=False)

temp = imports.mean(axis=0, level=['country', 'sector']).sum(axis=0, level='sector').mean(axis=1)
corr_sector_imports = imports.sum(axis=0, level=['sector', 'year']).groupby(['sector']).corr().unstack(level=1)[data_comb_cols]
corr_sector_imports[('mean', 'mean')] = temp 
corr_sector_imports = corr_sector_imports.sort_values(('mean', 'mean'), ascending=False)

temp = imports.mean(axis=0, level=['country', 'sector']).sum(axis=0, level='sector').mean(axis=1)
corr_sector_imports = imports.sum(axis=0, level=['sector', 'year']).groupby(['sector']).corr().unstack(level=1)[data_comb_cols]
corr_sector_imports[('mean', 'mean')] = temp 
corr_sector_imports = corr_sector_imports.sort_values(('mean', 'mean'), ascending=False)
 
####################
## Change in time ##
####################

years = range(2011, 2019)

# total ghg - all
change_total_ghg = total.groupby('year').sum().T
temp = cp.copy(change_total_ghg)
for year in years:
    change_total_ghg[year] = temp[year] / temp[year-1]
change_total_ghg = change_total_ghg.loc[:,2011:].T
for item in data_comb:
    change_total_ghg[item] = 0
    data0 = item.split(', ')[0]; data1 = item.split(', ')[1]
    change_total_ghg.loc[((change_total_ghg[data0] == 1) & (change_total_ghg[data1] == 1) |
                          (change_total_ghg[data0] > 1) & (change_total_ghg[data1] > 1) |
                          (change_total_ghg[data0] < 1) & (change_total_ghg[data1] < 1)), item] = 1
change_total_ghg = change_total_ghg[data_comb].sum().reset_index().rename(columns={'index':'data', 0:'Same_direction'})
change_total_ghg['Pct_same'] = change_total_ghg['Same_direction'] / len(years) * 100

# total ghg - imports
change_imports_ghg = imports.groupby('year').sum().T
temp = cp.copy(change_imports_ghg)
for year in years:
    change_imports_ghg[year] = temp[year] / temp[year-1]
change_imports_ghg = change_imports_ghg.loc[:,2011:].T
for item in data_comb:
    change_imports_ghg[item] = 0
    data0 = item.split(', ')[0]; data1 = item.split(', ')[1]
    change_imports_ghg.loc[((change_imports_ghg[data0] == 1) & (change_imports_ghg[data1] == 1) |
                          (change_imports_ghg[data0] > 1) & (change_imports_ghg[data1] > 1) |
                          (change_imports_ghg[data0] < 1) & (change_imports_ghg[data1] < 1)), item] = 1
change_imports_ghg = change_imports_ghg[data_comb].sum().reset_index().rename(columns={'index':'data', 0:'Same_direction'})
change_imports_ghg['Pct_same'] = change_imports_ghg['Same_direction'] / len(years) * 100

# sector x country 
change_all = (total + 0.01).unstack(['country', 'sector']).T
temp = cp.copy(change_all)
for year in years:
    change_all[year] = temp[year] / temp[year-1]
change_all = change_all.loc[:,2011:].unstack(level=['country', 'sector']).T
for item in data_comb:
    change_all[item] = 0
    data0 = item.split(', ')[0]; data1 = item.split(', ')[1]
    change_all.loc[((change_all[data0] == 1) & (change_all[data1] == 1) |
                          (change_all[data0] > 1) & (change_all[data1] > 1) |
                          (change_all[data0] < 1) & (change_all[data1] < 1)), item] = 1
change_all = change_all[data_comb].sum(level=['country', 'sector']).apply(lambda x: x/len(years) * 100)

# country 
change_country = (total + 0.01).sum(level=['country', 'year']).unstack(['country']).T
temp = cp.copy(change_country)
for year in years:
    change_country[year] = temp[year] / temp[year-1]
change_country = change_country.loc[:,2011:].unstack(level=['country']).T
for item in data_comb:
    change_country[item] = 0
    data0 = item.split(', ')[0]; data1 = item.split(', ')[1]
    change_country.loc[((change_country[data0] == 1) & (change_country[data1] == 1) |
                          (change_country[data0] > 1) & (change_country[data1] > 1) |
                          (change_country[data0] < 1) & (change_country[data1] < 1)), item] = 1
change_country = change_country[data_comb].sum(level=['country']).apply(lambda x: x/len(years) * 100)
  
# sector - all
change_sector_total = (total + 0.01).sum(level=['sector', 'year']).unstack(['sector']).T
temp = cp.copy(change_sector_total)
for year in years:
    change_sector_total[year] = temp[year] / temp[year-1]
change_sector_total = change_sector_total.loc[:,2011:].unstack(level=['sector']).T
for item in data_comb:
    change_sector_total[item] = 0
    data0 = item.split(', ')[0]; data1 = item.split(', ')[1]
    change_sector_total.loc[((change_sector_total[data0] == 1) & (change_sector_total[data1] == 1) |
                             (change_sector_total[data0] > 1) & (change_sector_total[data1] > 1) |
                             (change_sector_total[data0] < 1) & (change_sector_total[data1] < 1)), item] = 1
change_sector_total = change_sector_total[data_comb].sum(level=['sector']).apply(lambda x: x/len(years) * 100)

# sector - imports
change_sector_imports = (imports + 0.01).sum(level=['sector', 'year']).unstack(['sector']).T
temp = cp.copy(change_sector_imports)
for year in years:
    change_sector_imports[year] = temp[year] / temp[year-1]
change_sector_imports = change_sector_imports.loc[:,2011:].unstack(level=['sector']).T
for item in data_comb:
    change_sector_imports[item] = 0
    data0 = item.split(', ')[0]; data1 = item.split(', ')[1]
    change_sector_imports.loc[((change_sector_imports[data0] == 1) & (change_sector_imports[data1] == 1) |
                          (change_sector_imports[data0] > 1) & (change_sector_imports[data1] > 1) |
                          (change_sector_imports[data0] < 1) & (change_sector_imports[data1] < 1)), item] = 1
change_sector_imports = change_sector_imports[data_comb].sum(level=['sector']).apply(lambda x: x/len(years) * 100)

########################################
## Root Mean Squared Percentage Error ##
########################################

def rmspe(x, y):
    s = 0
    for i in range(len(x)):
        s += ((x[i]-y[i])/x[i]*100)**2
    e = np.sqrt(s/len(x))
    return e

ds = ['oecd', 'exio', 'figaro', 'gloria']
data_list = []
for item1 in ds:
    for item2 in ds:
        if item1 == item2:
            pass
        else:
            data_list.append([item1, item2])
            
# total ghg - all
temp = total.groupby('year').sum()
rmspe_total_ghg = pd.DataFrame()
for item in data_comb_cols:
    temp2 = pd.DataFrame(columns=['data', 'RMSPE1', 'RMSPE2'], index=[0])
    temp2.loc[0, 'data'] = item[0] + ', ' + item[1]
    temp2.loc[0, 'RMSPE1'] = rmspe(temp[item[0]].tolist(), temp[item[1]].tolist())
    temp2.loc[0, 'RMSPE2'] = rmspe(temp[item[1]].tolist(), temp[item[0]].tolist())
    rmspe_total_ghg = rmspe_total_ghg.append(temp2)
rmspe_total_ghg['rmspe'] = (rmspe_total_ghg['RMSPE1'] + rmspe_total_ghg['RMSPE2']) / 2

# total ghg - imports
temp = imports.groupby('year').sum()
rmspe_imports_ghg = pd.DataFrame()
for item in data_comb_cols:
    temp2 = pd.DataFrame(columns=['data', 'RMSPE1', 'RMSPE2'], index=[0])
    temp2.loc[0, 'data'] = item[0] + ', ' + item[1]
    temp2.loc[0, 'RMSPE1'] = rmspe(temp[item[0]].tolist(), temp[item[1]].tolist())
    temp2.loc[0, 'RMSPE2'] = rmspe(temp[item[1]].tolist(), temp[item[0]].tolist())
    rmspe_imports_ghg = rmspe_imports_ghg.append(temp2)
rmspe_imports_ghg['rmspe'] = (rmspe_imports_ghg['RMSPE1'] + rmspe_imports_ghg['RMSPE2']) / 2

# sector x country 
temp = (total + 0.01).unstack(['country', 'sector'])
temp.columns = temp.columns.swaplevel(0, 1).swaplevel(1, 2)
rmspe_all = pd.DataFrame()
for item in data_comb_cols:
    for c in temp.columns.levels[0]:
        for s in temp.columns.levels[1]:
            temp3 = temp[c][s]
            temp2 = pd.DataFrame(columns=['data', 'country', 'sector', 'RMSPE1', 'RMSPE2'], index=[0])
            temp2.loc[0, 'data'] = item[0] + ', ' + item[1]
            temp2.loc[0, 'country'] = c
            temp2.loc[0, 'sector'] = s
            temp2.loc[0, 'RMSPE1'] = rmspe(temp3[item[0]].tolist(), temp3[item[1]].tolist())
            temp2.loc[0, 'RMSPE2'] = rmspe(temp3[item[1]].tolist(), temp3[item[0]].tolist())
            rmspe_all = rmspe_all.append(temp2)
rmspe_all['rmspe'] = (rmspe_all['RMSPE1'] + rmspe_all['RMSPE2']) / 2
rmspe_all = rmspe_all.set_index(['data', 'country', 'sector'])[['rmspe']].unstack('data').droplevel(axis=1, level=0)

# country 
temp = (total + 0.01).sum(level=['country', 'year']).unstack(['country']).swaplevel(axis=1)
rmspe_country = pd.DataFrame()
for item in data_comb_cols:
    for c in temp.columns.levels[0]:
        temp3 = temp[c]
        temp2 = pd.DataFrame(columns=['data', 'country', 'RMSPE1', 'RMSPE2'], index=[0])
        temp2.loc[0, 'data'] = item[0] + ', ' + item[1]
        temp2.loc[0, 'country'] = c
        temp2.loc[0, 'RMSPE1'] = rmspe(temp3[item[0]].tolist(), temp3[item[1]].tolist())
        temp2.loc[0, 'RMSPE2'] = rmspe(temp3[item[1]].tolist(), temp3[item[0]].tolist())
        rmspe_country = rmspe_country.append(temp2)
rmspe_country['rmspe'] = (rmspe_country['RMSPE1'] + rmspe_country['RMSPE2']) / 2
rmspe_country = rmspe_country.set_index(['data', 'country' ])[['rmspe']].unstack('data').droplevel(axis=1, level=0)
  
# sector - all
temp = (total + 0.01).sum(level=['sector', 'year']).unstack(['sector']).swaplevel(axis=1)
rmspe_sector_total = pd.DataFrame()
for item in data_comb_cols:
    for c in temp.columns.levels[0]:
        temp3 = temp[c]
        temp2 = pd.DataFrame(columns=['data', 'sector', 'RMSPE1', 'RMSPE2'], index=[0])
        temp2.loc[0, 'data'] = item[0] + ', ' + item[1]
        temp2.loc[0, 'sector'] = c
        temp2.loc[0, 'RMSPE1'] = rmspe(temp3[item[0]].tolist(), temp3[item[1]].tolist())
        temp2.loc[0, 'RMSPE2'] = rmspe(temp3[item[1]].tolist(), temp3[item[0]].tolist())
        rmspe_sector_total = rmspe_sector_total.append(temp2)
rmspe_sector_total['rmspe'] = (rmspe_sector_total['RMSPE1'] + rmspe_sector_total['RMSPE2']) / 2
rmspe_sector_total = rmspe_sector_total.set_index(['data', 'sector' ])[['rmspe']].unstack('data').droplevel(axis=1, level=0)

# sector - imports
temp = (imports + 0.01).sum(level=['sector', 'year']).unstack(['sector']).swaplevel(axis=1)
rmspe_sector_imports = pd.DataFrame()
for item in data_comb_cols:
    for c in temp.columns.levels[0]:
        temp3 = temp[c]
        temp2 = pd.DataFrame(columns=['data', 'sector', 'RMSPE1', 'RMSPE2'], index=[0])
        temp2.loc[0, 'data'] = item[0] + ', ' + item[1]
        temp2.loc[0, 'sector'] = c
        temp2.loc[0, 'RMSPE1'] = rmspe(temp3[item[0]].tolist(), temp3[item[1]].tolist())
        temp2.loc[0, 'RMSPE2'] = rmspe(temp3[item[1]].tolist(), temp3[item[0]].tolist())
        rmspe_sector_imports = rmspe_sector_imports.append(temp2)
rmspe_sector_imports['rmspe'] = (rmspe_sector_imports['RMSPE1'] + rmspe_sector_imports['RMSPE2']) / 2
rmspe_sector_imports = rmspe_sector_imports.set_index(['data', 'sector' ])[['rmspe']].unstack('data').droplevel(axis=1, level=0)


##############
## Plot all ##
##############

### Correlations

# total and import ghg
plot_data = pd.DataFrame(corr_total_ghg).rename(columns={0:'total'}).join(pd.DataFrame(corr_imports_ghg).rename(columns={0:'imports'}))
plot_data = plot_data.T[data_comb_cols]
plot_data.columns = data_comb
plot_data = pd.DataFrame(plot_data.unstack())
order = plot_data.swaplevel(0).loc['imports'].sort_values(0, ascending=False).index.tolist()
plot_data = plot_data.reset_index().rename(columns={'level_0':'data', 'level_1':'ghg from', 0:'corr'})
plot_data['data'] = pd.Categorical(plot_data['data'], categories=order, ordered=True)
sns.scatterplot(data=plot_data.sort_values('corr', ascending=False), x='data', y='corr', hue='ghg from')
plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1)); plt.ylim(0, 1.1)
plt.show()


# country x sector total by emission decile
q = 5
for item in ['total', 'imports']:
    if item == 'total':
        plot_data = cp.copy(corr_all)
    elif item == 'imports':
        plot_data = cp.copy(corr_all).drop('United Kingdom')  
    plot_data = plot_data[data_comb_cols + [('mean', 'mean')]].fillna(0).sort_values(('mean', 'mean'), ascending=False)
    plot_data.columns = data_comb + ['mean']
    plot_data['cumsum'] = plot_data['mean'].cumsum()
    plot_data['cumpct'] = plot_data['cumsum'] / plot_data['mean'].sum() * 100
    plot_data.loc[plot_data['cumpct'] > 100, 'cumpct'] = 100
    for i in range(q):
        pct0 = 100/q*i; pct1 = 100/q*(i+1)
        plot_data.loc[(plot_data['cumpct'] <= pct1) & (plot_data['cumpct'] > pct0), 'quantile'] = str(pct0) + '-' + str(pct1) + '%'
    plot_data = plot_data[data_comb + ['quantile', 'mean']].set_index(['mean', 'quantile'], append=True).stack().reset_index().rename(columns={'level_4':'data', 0:'corr'})
    temp = plot_data.groupby('data').mean()[['corr']].reset_index().rename(columns={'corr':'mean_corr'})
    plot_data = plot_data.merge(temp, on='data').sort_values(['mean', 'mean_corr'], ascending=False)
    # country label
    sns.boxplot(data=plot_data, x='quantile', y='corr', hue='data'); 
    plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1)); plt.ylim(-1.1, 1.1)
    plt.title(item)
    plt.show()
    # pair label
    sns.boxplot(data=plot_data, x='data', y='corr', hue='quantile'); 
    plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1)); plt.ylim(-1.1, 1.1)
    plt.title(item)
    plt.show()

# total
plot_data = corr_country[data_comb_cols + [('mean', 'mean')]]
plot_data.columns = data_comb + ['mean']
temp = plot_data[data_comb].mean(0).reset_index().rename(columns={0:'mean_corr', 'index':'data'})
plot_data = plot_data.set_index('mean', append=True).stack().reset_index().rename(columns={'level_2':'data', 0:'corr'}).merge(temp, on='data').sort_values(['mean', 'mean_corr'], ascending=False)
# country label
sns.scatterplot(data=plot_data, x='country', y='corr', hue='data'); 
plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1)); plt.ylim(-1.1, 1.1)
plt.show()
# pair label
sns.boxplot(data=plot_data, x='data', y='corr'); 
plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1)); plt.ylim(-1.1, 1.1)
plt.show()

# country
plot_data = corr_country[data_comb_cols + [('mean', 'mean')]]
plot_data.columns = data_comb + ['mean']
temp = plot_data[data_comb].mean(0).reset_index().rename(columns={0:'mean_corr', 'index':'data'})
plot_data = plot_data.set_index('mean', append=True).stack().reset_index().rename(columns={'level_2':'data', 0:'corr'}).merge(temp, on='data').sort_values(['mean', 'mean_corr'], ascending=False)
# country label
sns.scatterplot(data=plot_data, x='country', y='corr', hue='data'); 
plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1)); plt.ylim(-1.1, 1.1)
plt.show()
# pair label
sns.boxplot(data=plot_data, x='data', y='corr'); 
plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1)); plt.ylim(-1.1, 1.1)
plt.show()

# sector
plot_data = corr_sector_imports[data_comb_cols + [('mean', 'mean')]]
plot_data.columns = data_comb + ['mean']
temp = plot_data[data_comb].mean(0).reset_index().rename(columns={0:'mean_corr', 'index':'data'})
plot_data = plot_data.set_index('mean', append=True).stack().reset_index().rename(columns={'level_2':'data', 0:'corr'}).merge(temp, on='data').sort_values(['mean', 'mean_corr'], ascending=False)
# sector label
sns.boxplot(data=plot_data, x='sector', y='corr'); 
plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1)); plt.ylim(-1.1, 1.1)
plt.show()
# pair label
sns.boxplot(data=plot_data, x='data', y='corr'); 
plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1)); plt.ylim(-1.1, 1.1)
plt.show()



### Change over time

# total and import ghg
plot_data = change_total_ghg.rename(columns={'Pct_same':'total'}).merge(change_imports_ghg.rename(columns={'Pct_same':'imports'}), on='data')
plot_data = pd.DataFrame(plot_data.set_index('data')[['total', 'imports']].stack())
order = plot_data.swaplevel(0).loc['imports'].sort_values(0, ascending=False).index.tolist()
plot_data = plot_data.reset_index().rename(columns={'level_1':'ghg from', 0:'pct_same'})
plot_data['data'] = pd.Categorical(plot_data['data'], categories=order, ordered=True)
sns.scatterplot(data=plot_data.sort_values('data', ascending=False), x='data', y='pct_same', hue='ghg from')
plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1)); plt.ylim(-5, 105)
plt.show()


# country x sector total by emission decile
change_all['mean'] = corr_all[('mean', 'mean')]
q = 5
for item in ['total', 'imports']:
    if item == 'total':
        plot_data = cp.copy(change_all)
    elif item == 'imports':
        plot_data = cp.copy(change_all).drop('United Kingdom')  
    plot_data = plot_data[data_comb + ['mean']].fillna(0).sort_values('mean', ascending=False)
    plot_data['cumsum'] = plot_data['mean'].cumsum()
    plot_data['cumpct'] = plot_data['cumsum'] / plot_data['mean'].sum() * 100
    plot_data.loc[plot_data['cumpct'] > 100, 'cumpct'] = 100
    for i in range(q):
        pct0 = 100/q*i; pct1 = 100/q*(i+1)
        plot_data.loc[(plot_data['cumpct'] <= pct1) & (plot_data['cumpct'] > pct0), 'quantile'] = str(pct0) + '-' + str(pct1) + '%'
    plot_data = plot_data[data_comb + ['quantile', 'mean']].set_index(['mean', 'quantile'], append=True).stack().reset_index().rename(columns={'level_4':'data', 0:'change'})
    temp = plot_data.groupby('data').mean()[['change']].reset_index().rename(columns={'change':'mean_change'})
    plot_data = plot_data.merge(temp, on='data').sort_values(['mean', 'mean_change'], ascending=False)
    # country label
    sns.boxplot(data=plot_data, x='quantile', y='change', hue='data'); 
    plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1)); plt.ylim(-5, 105)
    plt.title(item)
    plt.show()
    # pair label
    sns.boxplot(data=plot_data, x='data', y='change', hue='quantile'); 
    plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1)); plt.ylim(-5, 105)
    plt.title(item)
    plt.show()


# country
change_country['mean'] = corr_country[('mean', 'mean')]
plot_data = change_country[data_comb + ['mean']]
plot_data.columns = data_comb + ['mean']
temp = plot_data[data_comb].mean(0).reset_index().rename(columns={0:'mean_change', 'index':'data'})
plot_data = plot_data.set_index('mean', append=True).stack().reset_index().rename(columns={'level_2':'data', 0:'change'}).merge(temp, on='data').sort_values(['mean', 'mean_change'], ascending=False)
# country label
sns.scatterplot(data=plot_data, x='country', y='change', hue='data', style='data'); 
plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1)); plt.ylim(-5, 105)
plt.show()

g = sns.FacetGrid(plot_data.loc[plot_data['country'] != 'United Kingdom'], col="data", col_wrap=3)
g.map(sns.scatterplot, "country", "change")
plt.xticks(rotation=90); plt.ylim(-5, 105)
plt.show()


fig, ax1 = plt.subplots()
sns.boxplot(data=plot_data, x='country', y='change'); 
plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1)); plt.ylim(-5, 105)
ax2 = ax1.twinx()
ax2.plot(plot_data[['country', 'mean']].drop_duplicates()['country'], plot_data[['country', 'mean']].drop_duplicates()['mean'])
ax2.set_ylabel('Carbon emissions')
plt.show()

temp = plot_data.loc[plot_data['country'] != 'United Kingdom']
fig, ax1 = plt.subplots()
sns.boxplot(data=temp, x='country', y='change'); 
plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1)); plt.ylim(-5, 105)
ax2 = ax1.twinx()
ax2.plot(temp[['country', 'mean']].drop_duplicates()['country'], temp[['country', 'mean']].drop_duplicates()['mean'])
ax2.set_ylabel('Carbon emissions'); ax1.set_ylabel('Pct same')
plt.savefig(plot_filepath + '_boxplot_UK_imported_emissions_country_bycountry.png', dpi=200, bbox_inches='tight')
plt.show()
# pair label
sns.boxplot(data=plot_data, x='data', y='change'); 
plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1)); plt.ylim(-5, 105)
plt.savefig(plot_filepath + '_boxplot_UK_imported_emissions_country_bydata.png', dpi=200, bbox_inches='tight')
plt.show()

# sector - total
change_sector_total['mean'] = corr_sector_total[('mean', 'mean')]
plot_data = change_sector_total[data_comb + ['mean']]
plot_data.columns = data_comb + ['mean']
temp = plot_data[data_comb].mean(0).reset_index().rename(columns={0:'mean_change', 'index':'data'})
plot_data = plot_data.set_index('mean', append=True).stack().reset_index().rename(columns={'level_2':'data', 0:'change'}).merge(temp, on='data').sort_values(['mean', 'mean_change'], ascending=False)
# sector label
fig, ax1 = plt.subplots()
sns.boxplot(ax=ax1, data=plot_data, x='sector', y='change'); 
plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1)); plt.ylim(-5, 105)
ax2 = ax1.twinx()
ax2.plot(plot_data[['sector', 'mean']].drop_duplicates()['sector'], plot_data[['sector', 'mean']].drop_duplicates()['mean'])
ax2.set_ylabel('Carbon emissions')
plt.show()
# pair label
sns.boxplot(data=plot_data, x='data', y='change'); 
plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1)); plt.ylim(-5, 105)
plt.show()

# sector - imports
change_sector_imports['mean'] = corr_sector_imports[('mean', 'mean')]
plot_data = change_sector_imports[data_comb + ['mean']]
plot_data.columns = data_comb + ['mean']
temp = plot_data[data_comb].mean(0).reset_index().rename(columns={0:'mean_change', 'index':'data'})
plot_data = plot_data.set_index('mean', append=True).stack().reset_index().rename(columns={'level_2':'data', 0:'change'}).merge(temp, on='data').sort_values(['mean', 'mean_change'], ascending=False)
# sector label
fig, ax1 = plt.subplots()
sns.boxplot(ax=ax1, data=plot_data, x='sector', y='change'); 
plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1)); plt.ylim(-5, 105)
ax2 = ax1.twinx()
ax2.plot(plot_data[['sector', 'mean']].drop_duplicates()['sector'], plot_data[['sector', 'mean']].drop_duplicates()['mean'])
ax2.set_ylabel('Carbon emissions')
plt.savefig(plot_filepath + '_boxplot_UK_imported_emissions_sector_bysector.png', dpi=200, bbox_inches='tight')
plt.show()
# pair label
sns.boxplot(data=plot_data, x='data', y='change'); 
plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1)); plt.ylim(-5, 105)
plt.savefig(plot_filepath + '_boxplot_UK_imported_emissions_sector_bydata.png', dpi=200, bbox_inches='tight')
plt.show()


### RMPSE

# total and import ghg
plot_data = rmspe_total_ghg.rename(columns={'rmspe':'total'}).merge(rmspe_imports_ghg.rename(columns={'rmspe':'imports'}), on='data')
plot_data = pd.DataFrame(plot_data.set_index('data')[['total', 'imports']].stack())
order = plot_data.swaplevel(0).loc['imports'].sort_values(0, ascending=True).index.tolist()
plot_data = plot_data.reset_index().rename(columns={'level_1':'ghg from', 0:'rmspe'})
plot_data['data'] = pd.Categorical(plot_data['data'], categories=order, ordered=True)
sns.scatterplot(data=plot_data.sort_values('data', ascending=True), x='data', y='rmspe', hue='ghg from')
plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1)); plt.ylim(-5, 105)
plt.show()


# country x sector total by emission decile
rmspe_all['mean'] = corr_all[('mean', 'mean')]
q = 5
for item in ['total', 'imports']:
    if item == 'total':
        plot_data = cp.copy(rmspe_all)
    elif item == 'imports':
        plot_data = cp.copy(rmspe_all).drop('United Kingdom')  
    plot_data = plot_data[data_comb + ['mean']].fillna(0).sort_values('mean', ascending=True)
    plot_data['cumsum'] = plot_data['mean'].cumsum()
    plot_data['cumpct'] = plot_data['cumsum'] / plot_data['mean'].sum() * 100
    plot_data.loc[plot_data['cumpct'] > 100, 'cumpct'] = 100
    for i in range(q):
        pct0 = 100/q*i; pct1 = 100/q*(i+1)
        plot_data.loc[(plot_data['cumpct'] <= pct1) & (plot_data['cumpct'] > pct0), 'quantile'] = str(pct0) + '-' + str(pct1) + '%'
    plot_data = plot_data[data_comb + ['quantile', 'mean']].set_index(['mean', 'quantile'], append=True).stack().reset_index().rename(columns={'level_4':'data', 0:'rmspe'})
    temp = plot_data.groupby('data').mean()[['rmspe']].reset_index().rename(columns={'rmspe':'mean_rmspe'})
    plot_data = plot_data.merge(temp, on='data').sort_values(['mean', 'mean_rmspe'], ascending=True)
    # country label
    sns.boxplot(data=plot_data, x='quantile', y='rmspe', hue='data', showfliers=False); 
    plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1)); plt.yscale('log'); #plt.ylim(-5, 105)
    plt.title(item)
    plt.show()
    # pair label
    sns.boxplot(data=plot_data, x='data', y='rmspe', hue='quantile', showfliers=False); 
    plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1)); plt.yscale('log') #plt.ylim(-5, 105)
    plt.title(item)
    plt.show()

# country
rmspe_country['mean'] = corr_country[('mean', 'mean')]
plot_data = rmspe_country[data_comb + ['mean']]
plot_data.columns = data_comb + ['mean']
temp = plot_data[data_comb].mean(0).reset_index().rename(columns={0:'mean_rmspe', 'index':'data'})
plot_data = plot_data.set_index('mean', append=True).stack().reset_index().rename(columns={'level_2':'data', 0:'rmspe'}).merge(temp, on='data').sort_values(['mean', 'mean_rmspe'], ascending=True)
# country label
sns.scatterplot(data=plot_data, x='country', y='rmspe', hue='data'); 
plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1));# plt.ylim(-5, 105)
plt.show()
sns.boxplot(data=plot_data, x='country', y='rmspe'); 
plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1)); #plt.ylim(-5, 105)
plt.show()
# pair label
sns.boxplot(data=plot_data, x='data', y='rmspe'); 
plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1)); #plt.ylim(-5, 105)
plt.show()

# sector - total
rmspe_sector_total['mean'] = corr_sector_total[('mean', 'mean')]
plot_data = rmspe_sector_total[data_comb + ['mean']]
plot_data.columns = data_comb + ['mean']
temp = plot_data[data_comb].mean(0).reset_index().rename(columns={0:'mean_rmspe', 'index':'data'})
plot_data = plot_data.set_index('mean', append=True).stack().reset_index().rename(columns={'level_2':'data', 0:'rmspe'}).merge(temp, on='data').sort_values(['mean', 'mean_rmspe'], ascending=True)
# sector label
sns.boxplot(data=plot_data, x='sector', y='rmspe'); 
plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1)); plt.yscale('log'); # plt.ylim(-5, 105)
plt.show()
# pair label
sns.boxplot(data=plot_data, x='data', y='rmspe', showfliers=False); 
plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1)); #plt.ylim(-5, 105)
plt.show()

# sector - imports
rmspe_sector_imports['mean'] = corr_sector_imports[('mean', 'mean')]
plot_data = rmspe_sector_imports[data_comb + ['mean']]
plot_data.columns = data_comb + ['mean']
temp = plot_data[data_comb].mean(0).reset_index().rename(columns={0:'mean_rmspe', 'index':'data'})
plot_data = plot_data.set_index('mean', append=True).stack().reset_index().rename(columns={'level_2':'data', 0:'rmspe'}).merge(temp, on='data').sort_values(['mean', 'mean_rmspe'], ascending=True)
# sector label
sns.boxplot(data=plot_data, x='sector', y='rmspe'); 
plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1)); plt.yscale('log'); # plt.ylim(-5, 105)
plt.show()
# pair label
sns.boxplot(data=plot_data, x='data', y='rmspe', showfliers=False); plt.yscale('log'); # plt.ylim(-5, 105)
plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1)); #plt.ylim(-5, 105)
plt.show()