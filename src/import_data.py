# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:31:51 2023

@author: geolki
"""

# use pymrio https://pymrio.readthedocs.io/en/latest/notebooks/autodownload.html

# conda install -c conda-forge pymrio

import pymrio
import os


mrio_data_path = 'O:/geolki/data/raw/MRIOs/'

db = ['EXIO3']

for item in db:
    newpath = mrio_data_path + item
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        

data = pymrio.download_exiobase3(storage_folder=mrio_data_path + 'EXIO3/', system="pxp", years=[2011, 2012])

