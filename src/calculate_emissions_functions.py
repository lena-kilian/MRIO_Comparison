# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:31:51 2023

@author: geolki
"""

import pandas as pd
import numpy as np

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


def indirect_footprint(Z, Y, stressor):

    x = make_x(Z, Y)
    L = make_L(Z, x)
    s = stressor.apply(lambda i: i/x).fillna(0).iloc[:,0]
    sL = np.dot(np.diag(s), L)
    sLy = np.dot(sL, Y)
    
    footprint = pd.DataFrame(sLy, index=Y.index, columns=Y.columns)
        
    return footprint



