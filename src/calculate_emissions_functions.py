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


def indirect_footprint_Z(Z, Y, stressor):

    x = make_x(Z, Y)
    L = make_L(Z, x)
    e = stressor.apply(lambda i: i/x).fillna(0).iloc[:, 0]
    eL = np.dot(np.diag(e), L)
    eLy = np.dot(eL, Y)
    
    footprint = pd.DataFrame(eLy, index=Y.index, columns=Y.columns)
        
    return footprint


def make_Z_from_S_U(S, U):
    
    Z = np.zeros(shape = (np.size(S, 0)+np.size(U, 0), np.size(S, 1)+np.size(U, 1)))
    
    Z[np.size(S, 0):, 0:np.size(U, 1)] = U
    Z[0:np.size(S, 0), np.size(U, 1):] = S
        
    return Z


def indirect_footprint_SUT(S, U, Y, stressor):
    # make column names
    su_idx = pd.MultiIndex.from_arrays([[x[0] for x in S.columns.tolist()] + [x[0] for x in U.columns.tolist()],
                                        [x[1] for x in S.columns.tolist()] + [x[1] for x in U.columns.tolist()]])
    u_cols = U.columns.tolist()
    y_cols = Y.columns
    
    # calculate emissions
    Z = make_Z_from_S_U(S, U) 
    
    bigY = np.zeros(shape = [np.size(Z, 0), np.size(Y, 1)])
    
    footprint = np.zeros(shape = bigY.shape).T
    
    bigY[np.size(S, 0):np.size(Z, 0), 0:] = Y 
    x = make_x(Z, bigY)
    L = make_L(Z, x)
    bigstressor = np.zeros(shape = [np.size(Z, 0), 1])
    bigstressor[:np.size(S, 0), 0] = np.array(stressor)
    e = np.sum(bigstressor, 1)/x
    eL = np.dot(e, L)
    
    for a in range(np.size(Y, 1)):
        footprint[a] = np.dot(eL, np.diag(bigY[:, a]))
        print(a)
    
    footprint = pd.DataFrame(footprint, index=y_cols, columns=su_idx)
    footprint = footprint[u_cols]
     
    return footprint
