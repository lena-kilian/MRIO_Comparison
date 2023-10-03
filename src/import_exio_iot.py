# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:31:51 2023

@author: geolki
"""

import zipfile
from sys import platform
import requests
import io


# set working directory
# make different path depending on operating system
if platform[:3] == 'win':
    wd = 'O://'
else:
    wd = r'/Volumes/a72/' 

# define filepaths
data_filepath = wd + 'UKMRIO_Data/data/'

years = range(2010, 2019)
    
################
## EXIOBASE 3 ##
################

# Global MRIO

exio_folder = wd + 'UKMRIO_Data/EXIOBASE/IOT'

for year in years:
    url = 'https://zenodo.org/record/5589597/files/IOT_' + str(year) + '_ixi.zip?download=1'

    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(exio_folder)