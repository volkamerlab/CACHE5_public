# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 18:05:22 2024

@author: HP
"""

import pandas as pd
from tqdm import tqdm
import numpy as np
import os

import warnings

# To suppress pandas warnings globally:
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


def create_matrix(file_path : str , para : str , end_path : str):
    """
    Create Matrix
        Rows : LIG_ID
        Cols : GENE_ID

    Parameters
    ----------
    file_path : str

    para : str
        KD, KI etc.
        
    end_path : str

    Returns
    -------
    None.

    """
    df = pd.read_csv(file_path,
                     sep = '\t')
    
    table = df[['INTERACT' , para]]
    table.dropna(subset = [para] , inplace = True)
    table = table[table[para] <= 10000]
    
    grouped = table.groupby(['INTERACT'])
    grouped_median = grouped[para].median()
    
    table = grouped_median.to_frame()
    table.reset_index(inplace = True)
    table[['LIG_ID' , 'GENE_ID']] = table['INTERACT'].str.split('<>' , expand = True)
    table.drop(['INTERACT'], inplace = True , axis = 1)
    table = table[['LIG_ID' , 'GENE_ID' , para]]
    table[para] = table[para].astype(np.float64)
    
    
    matrix = table.pivot(index = 'LIG_ID',
                         columns = 'GENE_ID',
                         values = para)
    
    matrix.to_csv(end_path , sep = '\t')
    
    print(f'{para} DONE \n\n')


def bdb_main7():
    
    coeff = ['IC50' , 'KD' , 'EC50' , 'KON' , 'KOFF' , 'KI']
    
    mutN_final_int_path = 'processed/no_muts/mN_clean_interactions/BDB_mN_{}.tsv'
    mutY_final_int_path = 'processed/with_muts/mY_clean_interactions'
    
    mutN_matrix_path = 'processed/no_muts/mN_interact_matrix/BDB_mN_{}_Matrix.tsv'
    mutY_matrix_path = 'processed/with_muts/mY_interact_matrix/BDB_mY_{para}_Matrix.tsv'
    
    
    if not os.path.exists('processed/no_muts/mN_interact_matrix'):
        os.makedirs('processed/no_muts/mN_interact_matrix')
        print('Folder "processed/no_muts/mN_interact_matrix" created')
        
        
    if not os.path.exists('processed/with_muts/mY_interact_matrix'):
        os.makedirs('processed/with_muts/mY_interact_matrix')
        print('Folder "processed/with_muts/mY_interact_matrix" created')
    
    print('Creating matrices')
    for para in tqdm(coeff):
        
        # With No mutations
        file_path = mutN_final_int_path.format(para)
        end_path = mutN_matrix_path.format(para)
        
        if os.path.exists(end_path):
            print(f'{end_path} already exists. SKipping...')
        
        else:
            create_matrix(file_path, para, end_path)
        
            # With Yes Mutations
            file_path = mutY_final_int_path.format(para)
            end_path = mutY_matrix_path.format(para)
        
            create_matrix(file_path, para, end_path)
        
    print('Done')
