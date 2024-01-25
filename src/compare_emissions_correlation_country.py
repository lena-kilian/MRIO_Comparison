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
from sklearn.cluster import KMeans
import geopandas as gpd

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
    temp_oecd = pd.DataFrame(co2_all['oecd'][year].sum(axis=1, level=0).sum(axis=0)).rename(columns={0:'oecd'})
    temp_figaro = pd.DataFrame(co2_all['figaro'][year].sum(axis=1, level=0).sum(axis=0)).rename(columns={0:'figaro'})
    temp_gloria = pd.DataFrame(co2_all['gloria'][year].sum(axis=1, level=0).sum(axis=0)).rename(columns={0:'gloria'})
    temp_exio = pd.DataFrame(co2_all['exio'][year].sum(axis=1, level=0).sum(axis=0)).rename(columns={0:'exio'})
    
    temp = temp_oecd.join(temp_figaro).join(temp_gloria).join(temp_exio)
    
    temp['year'] = year
    
    summary = summary.append(temp.reset_index())
    
summary = summary.rename(columns={'index':'country'}).set_index(['country', 'year'])
    
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


