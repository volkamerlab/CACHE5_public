# -*- coding: utf-8 -*-
"""
Created on Sun May  5 12:16:29 2024

@author: HP
"""

import pandas as pd
from tqdm import tqdm
import os

def conversion(row):
    unit_type = row['UNIT']
    value = row['VALUE']
    
    match unit_type:
        case 'M':
            return(value*(10**9))
        case 'uM':
            return(value*(10**3))
        case 'um':
            return(value*(10**3))
        case 'mM':
            return(value*(10**6))
        case _:
            return value
        

def glass_main6():
    
    dir_path = 'processed/old/interactions'
    coeff = ['IC50' , 'EC50' , 'KD' , 'KI']
    
    for i in coeff:
        
        if os.path.exists(f'{dir_path}/GLASS_{i}.tsv'):
            print(f'{dir_path}/GLASS_{i}.tsv already exists. Skipping...')
            
        else:
            file_list = os.listdir(f'{dir_path}/{i}')
            final_df = pd.DataFrame()
            
            print(f'Concatenating files for {i}')
            for file in file_list:
                
                path = f'{dir_path}/{i}/{file}'
                
                try:
                    df = pd.read_csv(path , sep = '\t')
                except:
                    continue
                
                final_df = pd.concat([final_df , df])
            print('Done')
            
            print('Converting units to nM')
            good_unit = ['M' , 'nM' , '/nM' , 'um' , 'mM' , 'uM']
            final_df = final_df[final_df['UNIT'].isin(good_unit)]
            
            final_df['VALUE'] = final_df.apply(conversion, axis = 1)
            final_df['UNIT'] = 'nM'
            
            final_df.to_csv(f'{dir_path}/GLASS_{i}.tsv', sep = '\t')
            print('Done')
