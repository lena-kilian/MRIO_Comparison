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
    s = stressor.apply(lambda i: i/x).fillna(0).iloc[:, 0]
    sL = np.dot(np.diag(s), L)
    sLy = np.dot(sL, Y)
    
    footprint = pd.DataFrame(sLy, index=Y.index, columns=Y.columns)
        
    return footprint


def make_Z_from_S_U(S, U):
    
    Z = np.zeros(shape = (np.size(S, 0)+np.size(U, 0), np.size(S, 1)+np.size(U, 1)))
    
    Z[np.size(S, 0):, 0:np.size(U, 1)] = U
    Z[0:np.size(S, 0), np.size(U, 1):] = S
        
    return Z


def indirect_footprint_SUT(S, U, Y, stressor):
    
    temp = np.zeros(shape = np.size(Y, 1))
    Z = make_Z_from_S_U(S, U) 
    
    bigY = np.zeros(shape = [np.size(Y, 0)*2, np.size(Y, 1)])
    bigY[np.size(Y, 0):np.size(Y, 0)*2, 0:] = Y     
    x = make_x(Z, bigY)
    L = make_L(Z, x)
    bigstressor = np.zeros(shape = [np.size(Y, 0)*2, 1])
    bigstressor[:np.size(Y, 0), 0] = np.array(stressor)
    e = np.sum(bigstressor, 1)/x
    eL = np.dot(e, L)
    for a in range(np.size(Y, 1)):
        temp[a] = np.dot(eL, bigY[:, a])
    footprint = temp
      
    return footprint
