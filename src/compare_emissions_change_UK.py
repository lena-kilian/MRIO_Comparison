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

country = 'United Kingdom'

total = pd.DataFrame()
for year in years:
    temp_oecd = pd.DataFrame(co2_all['oecd'][year][[country]].sum(axis=1, level=0)).rename(columns={country:'oecd'})
    temp_figaro = pd.DataFrame(co2_all['figaro'][year][[country]].sum(axis=1, level=0)).rename(columns={country:'figaro'})
    temp_gloria = pd.DataFrame(co2_all['gloria'][year][[country]].sum(axis=1, level=0)).rename(columns={country:'gloria'})
    temp_exio = pd.DataFrame(co2_all['exio'][year][[country]].sum(axis=1, level=0)).rename(columns={country:'exio'})
    
    temp = temp_oecd.join(temp_figaro).join(temp_gloria).join(temp_exio)
    
    temp['year'] = year
    
    total = total.append(temp.reset_index())
    
total = total.rename(columns={'level_0':'country', 'level_1':'sector'}).set_index(['country', 'sector', 'year']).fillna(0)

domestic = total.loc[country, :]
imports = total.drop(country)

for item in ['total', 'domestic', 'imports']:
    data = eval(item).reset_index().groupby(['year']).sum()[datasets]
    data.plot()
    plt.title(country + ' ' + item)
    plt.savefig(plot_filepath + country + '_lineplot_country_emissions_' + item + '.png', dpi=200, bbox_inches='tight')


for pair in data_comb:
    sns.scatterplot(data=total, x=pair.split(', ')[0], y=pair.split(', ')[0], hue='country', style='year', legend=False)
    plt.yscale('log'); plt.xscale('log')
    plt.title('Total: ' + pair)
    plt.show()
    
    sns.scatterplot(data=imports, x=pair.split(', ')[0], y=pair.split(', ')[0], hue='country', style='year', legend=False)
    plt.yscale('log'); plt.xscale('log')
    plt.title('Imports: ' + pair)
    plt.show()
    
#####################
## Change in trend ##
#####################

# by country (of production)
data_country = total.sum(level=['country', 'year']).unstack('year').stack(level=0) + 0.01
change = data_country[years[1:]]
for year in years[1:]:
    change[year] = data_country[year] / data_country[year - 1]
    
change = change.reset_index().rename(columns={'level_1':'dataset'})

change3 = change.groupby(['country']).describe().stack(level=0)[['min', 'max', 'mean']]
change3['range'] = change3['max'] - change3['min']
change3['Same_direction'] = False
change3.loc[((change3['min']>1) & (change3['max']>1) |
             (change3['min']<1) & (change3['max']<1) |
             (change3['min']==1) & (change3['max']==1)), 'Same_direction'] = True

ghg = pd.DataFrame(total.sum(level=['country', 'year']).mean(axis=1)).rename(columns={0:'mean_ghg'}).apply(lambda x: pd.to_numeric(x, errors='coerce'))
ghg.loc[ghg['mean_ghg'] < 0, 'mean_ghg'] = 0
data_country = change3[['Same_direction', 'range']].reset_index()
data_country = data_country.set_index(['country', 'year']).join(ghg).reset_index().sort_values('mean_ghg', ascending=False)

# data_country = data_country.drop('mean_ghg', axis=1).join(data_country.groupby('country').mean()[['mean_ghg']])
data_country = data_country.set_index('country').reset_index().sort_values('mean_ghg', ascending=False)
sns.boxplot(data=data_country, x='country', y='range'); plt.legend(bbox_to_anchor=(1,1)); plt.xticks(rotation=90); plt.show()
sns.boxplot(data=data_country, x='country', y='range'); plt.legend(bbox_to_anchor=(1,1)); plt.xticks(rotation=90); plt.ylim(0, 1); plt.show()

sns.scatterplot(data=data_country, x='range', y='mean_ghg', hue='country'); plt.yscale('log'); plt.xscale('log'); plt.legend(bbox_to_anchor=(1,1))

# by sector
data_sector = total.sum(level=['sector', 'year']).unstack('year').stack(level=0) + 0.01
change = data_sector[years[1:]]
for year in years[1:]:
    change[year] = data_sector[year] / data_sector[year - 1]
    
change = change.reset_index().rename(columns={'level_1':'dataset'})

change3 = change.groupby(['sector']).describe().stack(level=0)[['min', 'max', 'mean']]
change3['range'] = change3['max'] - change3['min']
change3['Same_direction'] = False
change3.loc[((change3['min']>1) & (change3['max']>1) |
             (change3['min']<1) & (change3['max']<1) |
             (change3['min']==1) & (change3['max']==1)), 'Same_direction'] = True

ghg = pd.DataFrame(total.sum(level=['sector', 'year']).mean(axis=1)).rename(columns={0:'mean_ghg'}).apply(lambda x: pd.to_numeric(x, errors='coerce'))
ghg.loc[ghg['mean_ghg'] < 0, 'mean_ghg'] = 0
data_sector = change3[['Same_direction', 'range']].reset_index()
data_sector = data_sector.set_index(['sector', 'year']).join(ghg).reset_index().sort_values('mean_ghg', ascending=False)

data_sector = data_sector.set_index('sector').drop('mean_ghg', axis=1).join(data_sector.groupby('sector').mean()[['mean_ghg']]).reset_index().sort_values('mean_ghg', ascending=False)
data_sector['sector_short'] = data_sector['sector'].str[:10]
sns.boxplot(data=data_sector, x='sector_short', y='range'); plt.legend(bbox_to_anchor=(1,1)); plt.xticks(rotation=90); plt.show()
sns.boxplot(data=data_sector, x='sector_short', y='range'); plt.legend(bbox_to_anchor=(1,1)); plt.xticks(rotation=90); plt.ylim(0, 1); plt.show()

# Compare pariwise

change5 = change.set_index(['country', 'dataset']).unstack('dataset').stack('year')
change5.columns = pd.MultiIndex.from_arrays([['dataset']*len(datasets), change5.columns.tolist()])

for pair in data_comb:
    ds0 = pair.split(', ')[0]; ds1 = pair.split(', ')[1]
    change5[('Same_direction', ds0 + '_' + ds1)] = False
    change5.loc[((change5[('dataset', ds0)]>1) & (change5[('dataset', ds1)]>1) |
                 (change5[('dataset', ds0)]<1) & (change5[('dataset', ds1)]<1) |
                 (change5[('dataset', ds0)]==1) & (change5[('dataset', ds1)]==1)), 
                ('Same_direction', ds0 + '_' + ds1)] = True
    change5[('Abs_diff', ds0 + '_' + ds1)] = np.abs(change5[('dataset', ds0)] - change5[('dataset', ds1)])
    
count = change5[['Same_direction']].stack(level=1).reset_index().rename(columns={'level_2':'dataset'}).groupby(['country', 'dataset', 'Same_direction']).count().unstack('Same_direction').droplevel(axis=1, level=0).fillna(0)
count['pct_same'] = count[True] / (count[True] + count[False])*100
change6 = change5[['Abs_diff']].unstack('country').describe().T[['min', 'max', 'mean', 'std']].droplevel(axis=0, level=0)
change6.index.names = ['dataset', 'country']

change6 = change6.join(count[['pct_same']])

temp = change5['Abs_diff'].stack().reset_index().rename(columns={'level_2':'dataset', 0:'Abs_diff'})
order = temp.groupby('country').mean().sort_values('Abs_diff', ascending=True)
temp = temp.set_index(['country', 'year']).loc[order.index.tolist()].reset_index()
fig, ax = plt.subplots(figsize=(25,5))
sns.boxplot(ax=ax, data=temp, x='country', y='Abs_diff', hue='dataset', showfliers=False); 
plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1)); 
plt.savefig(plot_filepath + 'UK_boxplot_country_difference.png', dpi=200, bbox_inches='tight')

total_diff = order[['Abs_diff']]
total_diff['Under'] = 'Other'
for i in [50, 20, 10, 5, 1]:
    total_diff.loc[total_diff['Abs_diff'] <= i/100, 'Under'] = str(i) + '%'

sns.barplot(data=change6.reset_index(), x='country', y='pct_same', hue='dataset'); plt.xticks(rotation=90); plt.show()

change_country = change6.mean(axis=0, level='country').sort_values('pct_same', ascending=False)
change_data = change6.mean(axis=0, level='dataset').sort_values('pct_same', ascending=False)

sns.boxplot(data=change6.swaplevel(axis=0).loc[change_country.index.tolist()].reset_index(), x='country', y='pct_same'); plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1));
plt.savefig(plot_filepath + 'UK_boxplot_country_pctsame_bycountry.png', dpi=200, bbox_inches='tight')

sns.boxplot(data=change6.loc[change_data.index.tolist()].reset_index(), x='dataset', y='pct_same'); plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1)); plt.show()
plt.savefig(plot_filepath + 'UK_boxplot_country_pctsame_bydatapair.png', dpi=200, bbox_inches='tight')

sns.scatterplot(data=change6.swaplevel(axis=0).loc[change_country.index.tolist()].reset_index(), x='country', y='pct_same', hue='dataset'); plt.xticks(rotation=90); plt.legend(bbox_to_anchor=(1,1)); plt.show()
plt.savefig(plot_filepath + 'UK_scatterlot_country_pctsame.png', dpi=200, bbox_inches='tight')


 
#################
## Correlation ##
#################
    
correlation = pd.DataFrame();
for country in summary.index.levels[0].tolist():
    temp = summary.loc[country,:].corr()
    temp['country'] = country
    correlation = correlation.append(temp.set_index('country', append=True))
    
correlation = correlation.stack().reset_index()
correlation['combo'] = correlation['level_0'] + ', ' + correlation['level_2']
correlation = correlation.loc[correlation['combo'].isin(data_comb) == True]
correlation = correlation.set_index(['country', 'combo']).rename(columns={0:'corr'})[['corr']]


corr_summary_country = correlation.mean(axis=0, level=0).sort_values('corr', ascending = False)
corr_summary_country.plot(kind='bar')
plt.savefig(plot_filepath + 'barplot_country_correlation_bydatapair.png', dpi=200, bbox_inches='tight')

corr_summary_combo = correlation.mean(axis=0, level=1).sort_values('corr', ascending = False)
corr_summary_combo.plot(kind='bar')
plt.savefig(plot_filepath + 'barplot_country_correlation_bycountry.png', dpi=200, bbox_inches='tight')

#sns.scatterplot(data = correlation.reset_index(), x='combo',y='corr',  hue='country')
sns.boxplot(data = correlation.reset_index().sort_values('corr'), x='combo',y='corr'); plt.xticks(rotation=90)
plt.savefig(plot_filepath + 'boxplot_country_correlation_bydatapair.png', dpi=200, bbox_inches='tight')

correlation = correlation.unstack('combo')
correlation[('dataset', 'mean')] = correlation.mean(1)
correlation = correlation.append(pd.DataFrame(correlation.mean(0)).T.rename(index={0:'mean'}))
correlation = correlation.sort_values(('dataset', 'mean'), ascending=False).T.sort_values('mean', ascending=False).T

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
plot = world.set_index('name').join(
    correlation.rename(index={'United States':'United States of America', 'Czech Republic':'Czechia'})).fillna(correlation.loc['mean', ('dataset', 'mean')])

for item in plot.loc[:, ('corr', 'oecd, gloria'):].columns:
    plot.plot(column=item, legend=True, edgecolor='black', linewidth=0.1, cmap='RdBu'); 
    plt.title(str(item)); 
    plt.savefig(plot_filepath + 'map_world_country_correlation.png', dpi=200, bbox_inches='tight')
    
    plot.plot(column=item, legend=True, edgecolor='black', linewidth=0.1, cmap='RdBu'); 
    plt.xlim(-15, 50); plt.ylim(30, 85); plt.title(str(item)); 
    plt.savefig(plot_filepath + 'map_europe_country_correlation.png', dpi=200, bbox_inches='tight')


# Import Kmeans Library
# use the elbow method
wcss = []
results = correlation['corr']
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init = 'k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(results)
    wcss.append(kmeans.inertia_)
# Plot the WCSS results
plt.plot(range(1,11), wcss); plt.title('The elbow method'); plt.xlabel('number of clusters'); plt.ylabel('WCSS')
plt.show()


for i in [5]:
    results = correlation['corr']
    # Apply K-means to petal data based on WCSS results
    kmeans = KMeans(n_clusters=i, init = 'k-means++', max_iter=300, n_init=10, random_state=0)
    
    # this will create an arry for the predicted clusters for the petal data
    y_kmeans = kmeans.fit_predict(results)
    results['cluster'] = ['Cluster ' + str(x+1) for x in y_kmeans]
    
    results['median'] = results.iloc[:, :-1].median(1)
    results = results.sort_values('median', ascending = False)
    results = results.set_index(['cluster', 'median'], append=True).stack().reset_index()
    
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.boxplot(ax=ax, data=results, x='level_0', y=0, hue='cluster', dodge=False); 
    plt.xticks(rotation=90); 
    plt.title('No. clusters: ' + str(i))
    plt.savefig(plot_filepath + 'boxplot_country_correlation_bycountry.png', dpi=200, bbox_inches='tight')
