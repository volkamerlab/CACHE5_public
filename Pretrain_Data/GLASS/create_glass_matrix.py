# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:37:13 2024

@author: HP
"""

import pandas as pd
import os

def glass_main8():
    
    folder_path = 'processed/interaction_tables'
    matrix_path = 'processed/interaction_matrix'
    
    if not os.path.exists(matrix_path):
        os.makedirs(matrix_path)
        print(f'Folder {matrix_path} created')
    
    
    file_in_folder = os.listdir(folder_path)
    
    for file in file_in_folder:
        if os.path.exists(f'{matrix_path}/{file}'):
            print(f'{matrix_path}/{file} already exists. Skipping...')
            
        else:
            print(f'Creating matrix from {folder_path}/{file}')
            df = pd.read_csv(f'{folder_path}/{file}', sep = '\t')
            
            matrix = df.pivot(index = 'LIG_ID',
                              columns = 'GENE_ID',
                              values = 'VALUE')
            
            matrix.to_csv(f'{matrix_path}/{file}',
                          sep = '\t')
            print('Done')