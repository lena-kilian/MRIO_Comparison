# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:31:51 2023

@author: geolki
"""

# use pymrio https://pymrio.readthedocs.io/en/latest/notebooks/autodownload.html

# conda activate mrio

import pymrio
import os
import zipfile
from sys import platform


# set working directory
# make different path depending on operating system
if platform[:3] == 'win':
    wd = 'O://'
else:
    wd = r'/Volumes/a72/' 

# define filepaths
data_filepath = wd + 'UKMRIO_Data/data/'
mrio_data_path = wd + 'geolki/data/raw/MRIOs/'

db = ['Gloria']

for item in db:
     newpath = wd + 'UKMRIO_Data/' + item
     if not os.path.exists(newpath):
         os.makedirs(newpath)
     newpath = wd + 'UKMRIO_Data/' + item + '/Main'
     if not os.path.exists(newpath):
         os.makedirs(newpath)

years = range(2010, 2019)
    
#############
## GLORIA ##
############

gloria_folder = wd + 'UKMRIO_Data/Gloria/Main'

# Global MRIO
for year in years:
    gloria_log = pymrio.download_gloria(storage_folder=gloria_folder, year=year, overwrite_existing=True)
    
    for item in os.listdir(gloria_folder): # loop through items in dir
        if item.endswith('.zip'): # check for ".zip" extension
            file_name = gloria_folder + '/' + os.path.abspath(item).split('\\')[-1] # get full path of files
            with zipfile.ZipFile(file_name) as file: # create zipfile object temporarily
                file.extractall(gloria_folder) # extract file to dir
                file.close() # close file
            
            os.remove(file_name) # delete zipped file
        



