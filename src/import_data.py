# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:31:51 2023

@author: geolki
"""

# use pymrio https://pymrio.readthedocs.io/en/latest/notebooks/autodownload.html

import pymrio
import os


mrio_data_path = 'O:/geolki/data/raw/MRIOs/'

db = ['Exiobase3', 'OECD']

for item in db:
    newpath = mrio_data_path + item
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        
exio = pymrio.download_exiobase3(storage_folder=mrio_data_path + 'Exiobase3/', system="pxp", years=[2011])
oecd = pymrio.download_oecd(storage_folder=mrio_data_path + 'OECD/', version="v2016", years=[2003, 2008])
