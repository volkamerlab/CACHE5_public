#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 18:11:59 2024

@author: vka24
"""

import pandas as pd
import os
from collections import Counter


def Ki_table(df_ki : pd.core.frame.DataFrame , ki_table_loc):
    """
    Creating interaction table
    
    1) Keeping interactions with units (nM)
    2) Keeping interactions with value <= 10,000
    3) Finding median for each interaction

    Parameters
    ----------
    df_ki : pd.core.frame.DataFrame
        DESCRIPTION.

    Returns
    -------
    None.

    """
    df = df_ki
    units = list(set(df['UNIT']))
    df = df[df['UNIT'] == 'nM']
    
    df = df[['LIG_ID' , 'GENE_ID' , 'VALUE' , 'UNIT']]
    df.dropna(subset = ['VALUE'], inplace = True)
    df = df[df['VALUE'] <= 10000]
    df['INTERACT'] = df['LIG_ID'] + '::' + df['GENE_ID']
    
    
    grouped = df.groupby(['INTERACT'])
    
    grouped_median = grouped['VALUE'].median()
    table = grouped_median.to_frame()
    table.reset_index(inplace = True)
    table[['LIG_ID' , 'GENE_ID']] = table['INTERACT'].str.split('::' , expand = True)
    table.drop(['INTERACT'], inplace = True , axis = 1)
    table = table[['LIG_ID' , 'GENE_ID' , 'VALUE']]
    
    table.to_csv(ki_table_loc,
                 sep = '\t',
                 index = False)

def kd_table(df_kd : pd.core.frame.DataFrame , kd_table_loc):
    """
    Creating interaction table
    
    1) Keeping interactions with units (nM)
    2) Keeping interactions with value <= 10,000
    3) Finding median for each interaction

    Parameters
    ----------
    df_kd : pd.core.frame.DataFrame
        DESCRIPTION.

    Returns
    -------
    None.

    """
    df = df_kd
    units = list(set(df['UNIT']))
    
    df = df[df['UNIT'] == 'nM']
    
    df = df[['LIG_ID' , 'GENE_ID' , 'VALUE' , 'UNIT']]
    df.dropna(subset = ['VALUE'], inplace = True)
    df = df[df['VALUE'] <= 10000]
    df['INTERACT'] = df['LIG_ID'] + '::' + df['GENE_ID']
    
    
    grouped = df.groupby(['INTERACT'])
    
    grouped_median = grouped['VALUE'].median()
    table = grouped_median.to_frame()
    table.reset_index(inplace = True)
    table[['LIG_ID' , 'GENE_ID']] = table['INTERACT'].str.split('::' , expand = True)
    table.drop(['INTERACT'], inplace = True , axis = 1)
    table = table[['LIG_ID' , 'GENE_ID' , 'VALUE']]
    
    table.to_csv(kd_table_loc,
                 sep = '\t',
                 index = False)

def EC50_table(df_ec50 : pd.core.frame.DataFrame , ec50_table_loc):
    """
    Creating interaction table
    
    1) Keeping interactions with units (nM)
    2) Keeping interactions with value <= 10,000
    3) Finding median for each interaction

    Parameters
    ----------
    df_ec50 : pd.core.frame.DataFrame
        DESCRIPTION.

    Returns
    -------
    None.

    """
    df = df_ec50
    units = list(set(df['UNIT']))
    
    df = df[df['UNIT'] == 'nM']
    
    df = df[['LIG_ID' , 'GENE_ID' , 'VALUE' , 'UNIT']]
    df.dropna(subset = ['VALUE'], inplace = True)
    df = df[df['VALUE'] <= 10000]
    df['INTERACT'] = df['LIG_ID'] + '::' + df['GENE_ID']
    
    
    grouped = df.groupby(['INTERACT'])
    
    grouped_median = grouped['VALUE'].median()
    table = grouped_median.to_frame()
    table.reset_index(inplace = True)
    table[['LIG_ID' , 'GENE_ID']] = table['INTERACT'].str.split('::' , expand = True)
    table.drop(['INTERACT'], inplace = True , axis = 1)
    table = table[['LIG_ID' , 'GENE_ID' , 'VALUE']]
    
    table.to_csv(ec50_table_loc,
                 sep = '\t',
                 index = False)
       
def IC50_table(df_ic50 : pd.core.frame.DataFrame , ic50_table_loc):
    """
    Creating interaction table
    
    1) Keeping interactions with units (nM)
    2) Keeping interactions with value <= 10,000
    3) Finding median for each interaction

    Parameters
    ----------
    df_ic50 : pd.core.frame.DataFrame
            DESCRIPTION.

    Returns
    -------
    None.

    """
    df = df_ic50
    units = list(set(df['UNIT']))
    
    df = df[df['UNIT'] == 'nM']
    
    df = df[['LIG_ID' , 'GENE_ID' , 'VALUE' , 'UNIT']]
    df.dropna(subset = ['VALUE'], inplace = True)
    df = df[df['VALUE'] <= 10000]
    df['INTERACT'] = df['LIG_ID'] + '::' + df['GENE_ID']
    
    
    grouped = df.groupby(['INTERACT'])
    
    grouped_median = grouped['VALUE'].median()
    table = grouped_median.to_frame()
    table.reset_index(inplace = True)
    table[['LIG_ID' , 'GENE_ID']] = table['INTERACT'].str.split('::' , expand = True)
    table.drop(['INTERACT'], inplace = True , axis = 1)
    table = table[['LIG_ID' , 'GENE_ID' , 'VALUE']]
    
    table.to_csv(ic50_table_loc,
                 sep = '\t',
                 index = False)
    
def glass_main7():
    
    ki_input_loc = 'processed/old/interactions/GLASS_KI.tsv'
    kd_input_loc = 'processed/old/interactions/GLASS_KD.tsv'
    ic50_input_loc = 'processed/old/interactions/GLASS_IC50.tsv'
    ec50_input_loc =  'processed/old/interactions/GLASS_EC50.tsv'


    ki_table_loc = "processed/interaction_tables/GLASS_KI.tsv"
    kd_table_loc = "processed/interaction_tables/GLASS_KD.tsv"
    ic50_table_loc = "processed/interaction_tables/GLASS_IC50.tsv"
    ec50_table_loc = "processed/interaction_tables/GLASS_EC50.tsv"
    
    if not os.path.exists('processed/interaction_tables'):
        os.makedirs('processed/interaction_tables')
        print('Folder "processed/interactions" created')
    
    
    df_ki = pd.read_csv(ki_input_loc , sep = '\t')
    df_kd = pd.read_csv(kd_input_loc , sep = '\t')
    df_ec50 = pd.read_csv(ec50_input_loc , sep = '\t')
    df_ic50 = pd.read_csv(ic50_input_loc , sep = '\t')
    
    
    if os.path.exists(ki_table_loc):
        print(f'{ki_table_loc} already exists. Skipping...' )
        
    else:
        print('Creating KI table')
        Ki_table(df_ki , ki_table_loc)
        print('Done')
        
    if os.path.exists(kd_table_loc):
        print(f'{kd_table_loc} already exists. Skipping...')
        
    else:
        print('Creating KD table')
        kd_table(df_kd , kd_table_loc)
        print('Done')
        
    if os.path.exists(ec50_table_loc):
        print(f'{ec50_table_loc} already exists. Skipping...')
        
    else:
        print('Creating EC50 table')        
        EC50_table(df_ec50 , ec50_table_loc)
        print('Done')
        
    if os.path.exists(ic50_table_loc):
        print(f'{ic50_table_loc} already exits. Skipping...')
        
    else:
        print('Creating IC50 table')
        IC50_table(df_ic50 , ic50_table_loc)
        print('Done')
