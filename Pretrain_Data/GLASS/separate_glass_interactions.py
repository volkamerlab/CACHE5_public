#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 17:37:39 2024

@author: vka24
"""
        
import pandas as pd
from tqdm import tqdm
from collections import Counter
import os


import warnings

# To suppress pandas warnings globally:
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


def glass_main5():
    
    coeff = ['IC50' , 'EC50' , 'KD' , 'KI']
    
    clean_int_loc = 'processed/GLASS_Interaction_Fixed.tsv'
    para_count_loc = 'processed/GLASS_Parameter_Count.tsv'
    sep_int_path = 'processed/old/interactions'
    
    
    if not os.path.exists(sep_int_path):
        os.makedirs(sep_int_path)
        print(f'Folder "{sep_int_path}" created')
        
    if not os.path.exists(f'{sep_int_path}/others'):
        os.makedirs(f'{sep_int_path}/others')
        print(f'Folder "{sep_int_path}/others created')
    
    
    # Read the interaction data
    df = pd.read_csv(clean_int_loc, sep = '\t')
    
    
    if os.path.exists(para_count_loc):
        print(f'{para_count_loc} already exists. Skipping...')
        
    else:
        print('Creating parameter count file')
        # Create a file containing the counts for each parameter
        para_count = dict(Counter(list(df['PARAMETER'])))
        
        count_df = pd.DataFrame(para_count , index = ['COUNT']).T.reset_index()
        
        count_df.sort_values(by = 'COUNT' , inplace = True , ascending = False)
        
        count_df.rename(columns = {'index' : 'PARAMETER'}, inplace = True)
        
        count_df.to_csv(para_count_loc,
                        sep = '\t')
        print('Done')
    
    
    # Create a separate file for each parameter
    parameter_list = list(set(df['PARAMETER']))
    
    print('Separating interactions based on interaction coefficients')
    for parameter in parameter_list:
        # Getting dataset for that parameter
        df_para = df[df['PARAMETER'] == parameter]
        
        for i in coeff:
            
            if not os.path.exists(f'{sep_int_path}/{i}'):
                os.makedirs(f'{sep_int_path}/{i}')
                print(f'Folder {sep_int_path}/{i} created')
            
            
            if i.lower() in parameter.lower():
                file_path = f'{sep_int_path}/{i}/GLASS_{parameter}.tsv'
                break
                
            else:
                file_path = f'{sep_int_path}/others/GLASS_{parameter}.tsv'

        
        if os.path.exists(file_path):
            print(f'{file_path} already exists. Skipping...')
            
        else:
            df_para.to_csv(file_path,
                           sep = '\t',
                           index = False)
    print('Done')