#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:44:55 2024

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


def create_ID(row):
    """
    Combine gene_id & mut_id. if mut_id = None
    returns only gene_id

    Returns
    -------
    ID : str

    """
    
    try:
        ID = row['GENE_ID'] + '::' + row['MUT_ID']
        return ID
    
    except TypeError:
        ID = row['GENE_ID']
        return ID


def bdb_main2():
    
    tar_loc = "processed/old/BDB_Target.tsv"
    tar_loc1 = 'processed/old/BDB_Target1.tsv'
    
    mutY_tar_loc = 'processed/with_muts/BDB_mY_Target_Clean.tsv'
    mutY_tar_id_loc = 'processed/with_muts/BDB_mY_Target_IDs.tsv'

    mutN_tar_loc = 'processed/no_muts/BDB_mN_Target_Clean.tsv'
    mutN_tar_id_loc = 'processed/no_muts/BDB_mN_Target_IDs.tsv'
    
    
    if not os.path.exists('processed/with_muts'):
        os.makedirs('processed/with_muts')
        print('Folder "processed/with_muts" created')
        
    if not os.path.exists('processed/no_muts'):
        os.makedirs('processed/no_muts')
        print('Folder "processed/no_muts" created')
    
    
    if os.path.exists(tar_loc1):
        print(f'{tar_loc1} already exists. Skipping...')
        
    else:
        df = pd.read_csv(tar_loc, sep = '\t')
        # ID contains GENE+Mutation
        df.drop_duplicates(subset = ['ID'], inplace = True)
        
        # Previous script already had the combined ID
        df[['TEMP' , 'MUT']] =  df['ID'].str.split('::' , expand = True)
        df.drop('TEMP', axis = 1 , inplace = True)
        
        
        # Rearranging for sanity
        df = df[['ID', 'UNIPROT_ID' , 'MUT' , 'NAME', 'SOURCE_ORGANISM',
                'PDB_ID', 'SWISSPROT_NAME', 'SWISSPROT_ENTRY',
                'SWISSPROT_ID', 'SWISSPROT_SECONDARY_ID',
                'SWISSPROT_ALT_ID', 'TREMBL_NAME',
                'TREMBL_ENTRY', 'TREMBL_PRIMARY_ID',
                'TREMBL_SECONDARY_ID', 'TREMBL_ALT_ID', 'NO_PROT_CHAIN',
                'BDB_CHAIN_SEQ']]
        
        
        print('Creating Gene Mutation IDs')
        # Creating unique mutation IDs
        mut_df = df['MUT'].to_frame().dropna(subset = ['MUT']).drop_duplicates(subset = ['MUT'])
        
        mut_df['TEMP'] = range(1 , len(mut_df)+1)
        mut_df['MUT_ID'] = 'M' + mut_df['TEMP'].astype(str)
        mut_df.drop('TEMP', axis = 1 , inplace = True)
        mut_df.set_index('MUT', inplace = True)
        
        df.insert(3 , 'MUT_ID' , df['MUT'].map(mut_df['MUT_ID']))
        print('Done')
        
        
        print('Creating Gene IDs')
        # Creating unique gene IDs
        gene_df = df['UNIPROT_ID'].to_frame().dropna(subset = ['UNIPROT_ID']).drop_duplicates(subset = ['UNIPROT_ID'])
        
        gene_df['TEMP'] = range(1 , len(gene_df)+1)
        gene_df['GENE_ID'] = 'G' + gene_df['TEMP'].astype(str)
        gene_df.drop('TEMP', axis = 1 , inplace = True)
        gene_df.set_index('UNIPROT_ID', inplace = True)
        
        df.insert(2 , 'GENE_ID' , df['UNIPROT_ID'].map(gene_df['GENE_ID']))
        print('Done')
        
        
        df.to_csv(tar_loc1, sep = '\t',
                  index = False)
    
    
    if os.path.exists(mutY_tar_loc) and os.path.exists(mutY_tar_id_loc):
        print(f'{mutY_tar_loc} and {mutY_tar_id_loc} already present. Skipping...')
        
    else:
        # Making 2 copies of target dataset.
        # 1st copy will have ID = GENE_ID+MUT_ID
        # 2nd copy will have ID = GENE_ID
        print('Creating Target file with IDs having Mutation')
        df1 = pd.read_csv(tar_loc1, sep = '\t')
        df1.rename(columns = {'ID' : 'COMBINED_ID'}, inplace = True)
 
        df1.insert(0 , 'ID' , df.apply(create_ID , axis = 1))
        df1.to_csv(mutY_tar_loc, sep = '\t', index = False)
    
        id_df1 = df1[['ID' , 'COMBINED_ID']]
        id_df1.to_csv(mutY_tar_id_loc, sep = '\t', index = False)
        print('Done')
    
    if os.path.exists(mutN_tar_loc) and os.path.exists(mutN_tar_id_loc):
        print(f'{mutN_tar_loc} and {mutN_tar_id_loc} already present. Skipping...')
        
    else:   
        print('Creating Target file with IDs without mutations')
        df2 = pd.read_csv(tar_loc1, sep = '\t')
        df2.rename(columns = {'ID' : 'COMBINED_ID'}, inplace = True)
        df2.insert(0 , 'ID' , df2['GENE_ID'])
        # File did not have duplicate ID [UNIPROT_ID + MUTATION]
        # but there are duplicate UNIPROT_ID
        df2.drop_duplicates(subset = ['ID'], inplace = True)
        df2.to_csv(mutN_tar_loc, sep = '\t', index = False)
        
        id_df2 = df2[['ID' , 'UNIPROT_ID']]
        id_df2.to_csv(mutN_tar_id_loc, sep = '\t', index = False)
        print('Done')





