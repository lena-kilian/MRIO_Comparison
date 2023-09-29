# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:31:51 2023

@author: geolki
"""

import pandas as pd
import numpy as np

def make_Z_from_S_U(S, U):
    
    Z = np.zeros(shape = (np.size(S, 0)+np.size(U, 0), np.size(S, 1)+np.size(U, 1)))
    
    Z[np.size(S, 0):, 0:np.size(U, 1)] = U
    Z[0:np.size(S, 0), np.size(U, 1):] = S
        
    return Z


def make_bigY(Y, S, U):
    
    bigY = np.zeros(shape = (np.size(S, 0)+np.size(U, 0), np.size(Y, 1)))
    
    bigY[np.size(S, 0):, :] = Y
    
    return bigY
    

def make_x(Z, Y):
    
    x = np.sum(Z, 1)+np.sum(Y, 1)
    x[x == 0] = 0.000000001
    
    return x


def make_L(Z, x):
    
    bigX = np.zeros(shape = (len(Z)))    
    bigX = np.tile(np.transpose(x), (len(Z), 1))
    A = np.divide(Z, bigX)    
    L = np.linalg.inv(np.identity(len(Z))-A)

    return L

def make_e(stressor, x):
    
    e = np.zeros(shape = (1, np.size(x)))
    e[0, 0:np.size(stressor)] = np.transpose(stressor)
    e = e/x


def calculate_footprint(wd, Z, Y, stressor, direct, indicator, years):
    footprint = {}
   
    for year in years:
        print(year)
        
        x = make_x(Z, Y)
        L = make_L(Z, x)
        bigstressor = np.zeros(shape = [np.size(Y[year], 0)*2, 1])
        bigstressor[0:np.size(Y[year], 0), :] = stressor[year]
        e = np.sum(bigstressor, 1)/x 
        eL = np.dot(np.diag(e), L)

       
        drct[0, i] = direct.loc['Consumer expenditure - not travel', year]*co2_props.loc[year, 'Gas prop']
        drct[1, i] = direct.loc['Consumer expenditure - not travel', year]*co2_props.loc[year, 'Liquid fuel prop']
        drct[2, i] = direct.loc['Consumer expenditure - not travel', year]*co2_props.loc[year, 'Solid fuel prop']
        drct[3, i] = direct.loc['Consumer expenditure - travel', year]
        
        defra_foot[str(year)+'_sic'] = pd.DataFrame(ccc, index = meta['sectors']['ind'], columns = meta['sectors']['prd'])
        defra_foot[str(year)+'_reg'] = pd.DataFrame(reg, index = meta['reg']['idx'], columns = Y[year].columns[0:np.size(Y[year], 1)-1])

    defra_foot['direct'] = pd.DataFrame(drct, index = ['Consumer expenditure gas', 'Consumer expenditure liquid fuel', 
                                             'Consumer expenditure solid fuel', 'Consumer expenditure travel'], 
                              columns = allyears)
    
    return defra_foot



