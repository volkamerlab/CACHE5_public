#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 20:12:16 2024

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


def bdb_main4():
    
    int_loc = "processed/old/BDB_Interaction.tsv"
    dupe_pcid_loc = 'processed/old/BDB_Ligand_DupePCID.tsv'
    
    mutN_lig_id_loc = 'processed/no_muts/BDB_mN_Ligand_IDs.tsv'
    mutN_tar_id_loc = 'processed/no_muts/BDB_mN_Target_IDs.tsv'
    mutN_int_loc = 'processed/no_muts/BDB_mN_Interaction_Clean.tsv'
    
    mutY_lig_id_loc = 'processed/with_muts/BDB_mY_Ligand_IDs.tsv'
    mutY_tar_id_loc = 'processed/with_muts/BDB_mY_Target_IDs.tsv'
    mutY_int_loc = 'processed/with_muts/BDB_mY_Interaction_Clean.tsv'
    
    
    print('Cleaning Interaction data')
    df = pd.read_csv(int_loc,
                      sep = '\t')
    
    # Remove interactions for which ligands have common PCIDs
    dupe_df = pd.read_csv(dupe_pcid_loc, sep = '\t')
    dupe_list = list(set(dupe_df['INCHI_KEY']))
    
    df = df[~df['INCHI_KEY'].isin(dupe_list)]
    df.reset_index(inplace = True , drop = True)
    print('Done')
    
    if os.path.exists(mutN_int_loc):
        print(f'{mutN_int_loc} already exists. Skipping...')
        
    else:
        print('Adding new gene & mutation IDs to respective rows')
        # Add IDs Without Mutations
        lig_id = pd.read_csv(mutN_lig_id_loc, sep = '\t')
        lig_id.set_index('INCHI_KEY', inplace = True)
        
        gene_id = pd.read_csv(mutN_tar_id_loc, sep = '\t')
        gene_id.set_index('UNIPROT_ID', inplace = True)
        print('Done')
        
        print('Create interaction data with IDs having mutations')
        df1 = df.copy(deep = True)
        df1.insert(1 , 'GENE_ID' , df1['UNIPROT_ID'].map(gene_id['ID']))
        df1.insert(0 , 'LIG_ID' , df1['INCHI_KEY'].map(lig_id['LIG_ID']))
        
        col_list = list(df1.columns)
        col_list.append(col_list.pop(1))
        col_list.append(col_list.pop(2))
        col_list.append(col_list.pop(2))
        
        df1 = df1[col_list]
        
        df1.to_csv(mutN_int_loc, sep = '\t',
                  index = False)
        print('Done')
    
    if os.path.exists(mutY_int_loc):
        print(f'{mutY_int_loc} already exists. Skipping...')
        
    else:
        print('Create interaction data with IDs not having mutations')
        # Add IDs with Mutations
        lig_id = pd.read_csv(mutY_lig_id_loc, sep = '\t')
        lig_id.set_index('INCHI_KEY', inplace = True)
        
        gene_id = pd.read_csv(mutY_tar_id_loc, sep = '\t')
        gene_id.set_index('COMBINED_ID', inplace = True)
        
        
        df2 = df.copy(deep = True)
        df2.insert(1 , 'GENE_ID' , df2['ID'].map(gene_id['ID']))
        df2.insert(0 , 'LIG_ID' , df2['INCHI_KEY'].map(lig_id['LIG_ID']))
        
        col_list = list(df2.columns)
        col_list.append(col_list.pop(1))
        col_list.append(col_list.pop(2))
        col_list.append(col_list.pop(2))
        
        df2 = df2[col_list]
        
        df2.to_csv(mutY_int_loc, sep = '\t',
                  index = False)
        print('Done')
