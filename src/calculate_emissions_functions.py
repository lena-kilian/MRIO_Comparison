# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:31:51 2023

@author: geolki
"""

import pandas as pd
import numpy as np
import os

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


def calculate_footprint(wd, Z, Y, stressor, direct, indicator, years, meta):
    footprint = {}

    tempregagg = np.zeros((meta['fd']['len_idx'],meta['reg']['len'])) 
    tempsecagg = np.zeros((meta['fd']['len_idx'],meta['sup_dd']['len_idx']))
    for r in range(0,meta['reg']['len']):
        tempregagg[r*meta['sup_dd']['len_idx']:(r+1)*meta['sup_dd']['len_idx'],r] = np.transpose(np.ones((meta['sup_dd']['len_idx'],1)))
        tempsecagg[r*meta['sup_dd']['len_idx']:(r+1)*meta['sup_dd']['len_idx'],:] = np.identity(meta['sup_dd']['len_idx'])
    
    regagg = np.zeros((2*meta['fd']['len_idx'],meta['reg']['len'])) 
    secagg = np.zeros((2*meta['fd']['len_idx'],meta['sup_dd']['len_idx']))
    prdagg = np.zeros((2*meta['fd']['len_idx'],meta['sup_dd']['len_idx']))
    regagg[0:meta['fd']['len_idx'],:] = tempregagg
    secagg[0:meta['fd']['len_idx'],:] = tempsecagg
    prdagg[meta['fd']['len_idx']:,:] = tempsecagg

    drct = np.zeros((4,len(years)))
   
    sic = np.zeros((meta['sup_dd']['len_idx'],len(years)))
    
    energy_filepath = wd + 'data/processed/uk energy/'
    co2_props = pd.read_excel(os.path.join(energy_filepath, 'UKenergy2023.xlsx'), sheet_name='co2_props', header = 0, index_col=0)
        
    for i,year in enumerate(years):
        print(year)
        bigY = np.zeros(shape = [np.size(Y[year],0)*2,np.size(Y[year],1)])
        bigY[np.size(Y[year],0):np.size(Y[year],0)*2,0:np.size(Y[year],1)] = Y[year]
        x = make_x(Z,bigY)
        L = make_L(Z,x)
        bigstressor = np.zeros(shape = [np.size(Y[year],0)*2,1])
        bigstressor[0:np.size(Y[year],0),:] = stressor[year]
        e = np.sum(bigstressor,1)/x 
        eL = np.dot(np.diag(e),L)
        reg = np.zeros((meta['reg']['len'],40))
        prd = np.zeros((meta['sup_dd']['len_idx'],40))
        ccc = np.zeros((len(Z[year]),meta['sup_dd']['len_idx']))
        ygg = np.dot(np.sum(bigY[:,0:np.shape(bigY)[1]-1],1),prdagg)
        
        for ysec in range (0,40):
            reg[:,ysec] = np.dot(np.dot(eL,bigY[:,ysec]),regagg)
            prd[:,ysec] = np.dot(np.dot(np.sum(eL,0),np.diag(bigY[:,ysec])),prdagg)
        ccc = np.dot(np.transpose(secagg),np.dot(np.dot(eL,np.diag(np.sum(bigY[:,0:-1],1))),prdagg))
        sic[:,i] = np.sum(prd,1)/ygg
       
        drct[0,i] = direct.loc['Consumer expenditure - not travel',year]*co2_props.loc[year,'Gas prop']
        drct[1,i] = direct.loc['Consumer expenditure - not travel',year]*co2_props.loc[year,'Liquid fuel prop']
        drct[2,i] = direct.loc['Consumer expenditure - not travel',year]*co2_props.loc[year,'Solid fuel prop']
        drct[3,i] = direct.loc['Consumer expenditure - travel',year]
        
        footprint[str(year)+'_sic'] = pd.DataFrame(ccc, index = meta['sectors']['ind'], columns = meta['sectors']['prd'])
        footprint[str(year)+'_reg'] = pd.DataFrame(reg, index = meta['reg']['idx'], columns = Y[year].columns[0:np.size(Y[year],1)-1])

    footprint['direct'] = pd.DataFrame(drct, index = ['Consumer expenditure gas','Consumer expenditure liquid fuel', 
                                             'Consumer expenditure solid fuel', 'Consumer expenditure travel'],
                              columns = years)
    footprint['sic_mult'] = pd.DataFrame(sic, index =  meta['sectors']['prd'], columns = years)
        
    return footprint



