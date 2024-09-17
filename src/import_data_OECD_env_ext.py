#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 13:46:00 2018

This code imports the LCF original survey data from the UKDA for use with the UKMRIO

@authors: Anne Owen & Lena Kilian 
"""

import pandas as pd
df = pd.DataFrame
import requests as rq


wd = 'O://ESCoE_Project/data/MRIO/'

years = list(range(2010, 2024))

industries = ('D01T02+D03+D05T06+D07T08+D09+D10T12+D13T15+D16+D17T18+D19+D20+D21+D22+D23+D24+D25+D26+D27+D28+D29+D30+D31T33+D35+D36T39+' +
              'D41T43+D45T47+D49+D50+D51+D52+D53+D55T56+D58T60+D61+D62T63+D64T66+D68+D69T75+D77T82+D84+D85+D86T88+D90T93+D94T96+D97T98')#.split('+')

countries =  ('AUS+AUT+BEL+CAN+CHL+COL+CRI+CZE+DNK+EST+FIN+FRA+DEU+GRC+HUN+ISL+IRL+ISR+ITA+JPN+KOR+LVA+LTU+LUX+MEX+NLD+NZL+NOR+POL+PRT+SVK+SVN+ESP+SWE+' + 
              'CHE+TUR+GBR+USA+ARG+BRA+BRN+BGR+KHM+CHN+HRV+CYP+IND+IDN+HKG+KAZ+LAO+MYS+MLT+MAR+MMR+PER+PHL+ROU+RUS+SAU+SGP+ZAF+TWN+THA+TUN+VNM+' +
              'ROW')#.split('+')

partners = ('WLD+OECD+AUS+AUT+BEL+CAN+CHL+CZE+DNK+EST+FIN+FRA+DEU+GRC+HUN+ISL+IRL+ISR+ITA+JPN+KOR+LVA+LTU+LUX+MEX+NLD+NZL+NOR+POL+PRT+SVK+SVN+ESP+' +
            'SWE+CHE+TUR+GBR+USA+NONOECD+ARG+BRA+BRN+BGR+KHM+CHN+COL+CRI+HRV+CYP+HKG+IND+IDN+KAZ+MYS+MLT+MAR+PER+PHL+ROU+RUS+SAU+SGP+ZAF+TWN+THA+TUN+' +
            'VNM+ROW+APEC+ASEAN+EASIA+EU15+EU28+EU13+EA19+EA12+G20+ZNAM+ZEUR+ZASI+ZSCA+ZOTH+DXD').split('+')

indicators = ('FD_CO2+PROD_CO2').split('+')

# get lookup for industy names
ind_lookup = pd.read_excel(wd + 'ICIO/Ed_2024/ReadMe_ICIO_extended.xlsx', sheet_name='Area_Activities', header=2)[['Code.1', 'Industry']].dropna(how='all')
ind_lookup.columns = ['IND', 'IND_name']

# import data from web
partner = partners[0]

data_full = pd.read_csv(wd + 'ICIO/Ed_2024/Env_extensions/OECD.SDD.NAD.SEEA,DSD_AEA@DF_AEA,1.0+all.csv')
data_full['TIME_PERIOD'] = data_full['TIME_PERIOD'].astype(str)

for c in data_full.columns:
    print(c, data_full[c].unique())


to_drop = ['Decimals', 'DECIMALS', 'UNIT_MULT', 'Time period', 'STRUCTURE', 'STRUCTURE_ID', 'STRUCTURE_NAME', 
           'ACTION', 'Observation value', 'Unit multiplier', 'MEASURE', 'Measure', 'UNIT_MEASURE', 'FREQ',
           'Frequency of observation', 'OBS_STATUS', 'Adjustment', 'ADJUSTMENT', 'ACTIVITY_SCOPE', 'SOURCE',
           'POLLUTANT', 'METHODOLOGY']
data = data_full.loc[
    (data_full['POLLUTANT'] == 'CO2') &
    #(data_full['Unit of measure'] == 'Tonnes') &
    (data_full['TIME_PERIOD'].isin([str(x) for x in list(years)]) == True)
    ].drop(to_drop, axis=1)

for c in data.columns:
    print(c, data[c].unique())
    
check = data[['Reference area', 'TIME_PERIOD']].drop_duplicates()
check['test'] = 1
check = check.set_index(['Reference area', 'TIME_PERIOD']).unstack('TIME_PERIOD')

'''
check2 = data[['Reference area']].drop_duplicates()
check2['check2'] = 1
check3 = data_full[['Reference area']].drop_duplicates()
check3['check3'] = 1

check1 = check3.merge(check2, on='Reference area', how='outer')

# Australia, Canada
check = data.iloc[:100, :]
    
        url = ('https://sdmx.oecd.org/public/rest/data/OECD.SDD.NAD.SEEA,DSD_AEA@DF_AEA,1.0/BEL+CAN+COL+CRI+CZE+DNK+EST+FIN+FRA+DEU+GRC+HUN+ISL+IRL+ITA+JPN+KOR+LVA+LTU+LUX+MEX+NLD+NZL+NOR+POL+PRT+SVK+SVN+ESP+SWE+CHE+TUR+GBR+USA+EU27_2020+OECD+WXOECD+BGR+HRV+CYP+IDN+KAZ+MLT+ROU+RUS+SRB+UKR+AUT+AUS...T.N.A01+A02+A03+A_X+C10T12+C13T15+C16T18+C16+C17+C18+C19+C20_21+C20+C21+C22_23+C22+C23+C24_25+C24+C25+C26+C27+C28+C29_30+C29+C30+C31_32+C31T33+C33+C_X+E36+E37T39+G45+G46+G47+G_X+H49+H50+H51+H52+H53+H_X+J58T60+J58+J59_60+J61+J62_63+J_X+K64+K65+K66+K_X+L68A+L68+M69_70+M69T71+M71+M72+M73T75+M73+M74_75+M_X+N77+N78+N79+N80T82+N_X+Q86+Q87_88+Q_X+R90T92+R93+R_X+S94+S95+S96+S_X+B+D+F+I+O+P+T+U+HH_TR+HH_HEAT+HH_OTH.CO2.RES+RES_ABR+TER+TER_NRES+ADJ_SD+LULUCF+RES_ABR_ATR+RES_ABR_FWTR+RES_ABR_LTR+RES_ABR_WTR+TER_NRES_ATR+TER_NRES_LTR+TER_NRES_WTR..?startPeriod=2010&endPeriod=2024&dimensionAtObservation=AllDimensions')
                
        raw = rq.get(url).content.decode('utf-8');
    
        raw = raw.split('concept="VAR"')
        for i in range(len(raw)):
            raw[i] = raw[i].split('><')
            
        data = pd.DataFrame(raw).iloc[:,1:].dropna(how='any', axis=1)
        data.columns = data.iloc[1, :].tolist()
        keep = []
        for x in data.columns:
            if '=' in x:
                keep.append(x)
            if 'ObsValue' in x:
                a = x
        data = data[keep].rename(columns={a:'Value concept="VALUE" value='}).iloc[1:, :]
        data.columns = [x.split('value=')[0].split('=')[1].replace('"', '').replace(' ', '') for x in data.columns.tolist()]
        data = data.apply(lambda x: x.str.split('value="').str[1].str.split('"').str[0])
        data['YEAR'] = year
        
        data = data.merge(ind_lookup, on='IND', how='left')
        
        filename = wd + 'ICIO/Ed_2024/Env_extensions/Extension_data_' + str(year) + '_' + indicator + '_' + partner + '.csv'
        data.to_csv(filename)
        print('saved ' + filename)
    
'''
            
