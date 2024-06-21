#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 14:45:39 2024

@author: vka24
"""

import numpy as np
import pandas as pd

import os

import warnings

# To suppress pandas warnings globally:
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)



def glass_main3():
    
    tar_loc = 'processed/old/GLASS_Target.tsv'
    
    clean_tar_loc = 'processed/GLASS_Target_Fixed.tsv'
    clean_tar_id_loc = 'processed/GLASS_Target_ID.tsv'
    
    
    if os.path.exists(clean_tar_loc) and os.path.exists(clean_tar_id_loc):
        print(f'{clean_tar_loc} & {clean_tar_id_loc} already exists. Skipping...')
        
    else:
        print('Creating Gene IDs')
        # Read target file
        df = pd.read_csv(tar_loc,
                         sep = '\t')
        
        # No difference
        df.drop_duplicates(subset = ['ID'], inplace = True)
        
        # Create new IDs for Genes
        df['TEMP'] = range(1 , len(df)+1)
        df.insert(0 , 'GENE_ID' , 'G' + df['TEMP'].astype(str))
        df.drop(['TEMP'] , inplace = True , axis = 1)
        
        df.to_csv(clean_tar_loc,
                  sep = '\t',
                  index = False)
        
        gene_df = df[['GENE_ID' , 'ID']]
        gene_df.to_csv(clean_tar_id_loc,
                       sep = '\t',
                       index = False)
        
        print('Done')