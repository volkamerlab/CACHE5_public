#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 14:50:22 2024

@author: vka24
"""

import pandas as pd

import os

import warnings

# To suppress pandas warnings globally:
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


def glass_main2():
    
    # Paths
    lig_loc = 'processed/old/GLASS_Ligand.tsv'
    
    clean_lig_id_loc = 'processed/GLASS_Ligand_ID.tsv'
    clean_lig_loc = 'processed/GLASS_Ligand_Fixed.tsv'
    
    
    if os.path.exists(clean_lig_loc) and os.path.exists(clean_lig_id_loc):
        print(f'{clean_lig_loc} & {clean_lig_id_loc} already exists. Skipping...')
        
    else:     
        print('Creating Ligand IDs')
        # Read ligand file
        df = pd.read_csv(lig_loc, sep = '\t')
        
        # No change
        df.drop_duplicates(subset = ['INCHI_KEY'], inplace = True)
        
        # Creating new ligand IDs
        df['TEMP'] = range(1 , len(df)+1)
        df.insert(0 , 'LIG_ID' , 'L' + df['TEMP'].astype(str))
        df.drop(['TEMP'] , inplace = True , axis = 1)
        
        df.to_csv(clean_lig_loc,
                  sep = '\t',
                  index = False)
        
        # Ligand ID file
        ligand_df = df[['LIG_ID' , 'INCHI_KEY']]
        ligand_df.to_csv(clean_lig_id_loc,
                         sep = '\t',
                         index = False)
        print('Done')
