#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 14:29:31 2024

@author: vka24
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import os

import warnings

# To suppress pandas warnings globally:
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


def separate_doi(df : pd.core.frame.DataFrame):
    """
    Separate DataFrame based on the presence of DOI
    Separation is exclusive

    Parameters
    ----------
    df : pd.core.frame.DataFrame
        Should have ['DOI'] column

    Returns
    -------
    doi_y : pd.core.frame.DataFrame
        Dataframe with entries containing DOI
    doi_n : pd.core.frame.DataFrame
        Dataframe with entries not having DOI

    """
    doi_y = df[~df['DOI'].isnull()]
    doi_n = df[df['DOI'].isnull()]
    
    return doi_y , doi_n

def separate_pat(df : pd.core.frame.DataFrame):
    """
    Separate DataFrame based on the presence of PATENT
    Separation is exclusive
    
    Parameters
    ----------
    df : pd.core.frame.DataFrame
        Should have ['PATETNT'] column

    Returns
    -------
    pat_y : pd.core.frame.DataFrame
        DataFrame with entries containing PATENT
    pat_n : pd.core.frame.DataFrame
        DataFrame with entries not having PATENT

    """
    pat_y = df[~df['PATENT'].isnull()]
    pat_n = df[df['PATENT'].isnull()]
    
    return pat_y , pat_n

def separate_pmid(df : pd.core.frame.DataFrame):
    """
    Separate DataFrame based on presence of PMID

    Parameters
    ----------
    df : pd.core.frame.DataFrame
        Should contain ['PMID'] column

    Returns
    -------
    pmid_y : pd.core.frame.DataFrame
        DataFrame with entries containing PMID
    pmid_n : pd.core.frame.DataFrame
        DataFrame with entries not having PMID

    """
    pmid_y = df[~df['PMID'].isnull()]
    pmid_n = df[df['PMID'].isnull()]
    
    return pmid_y , pmid_n

def check_nan_paid(df : pd.core.frame.DataFrame):
    """
    For entries with same INCHI_KEY, UNIPROT_ID,
    Interaction Value, Reference [DOI or PATENT],
    
    Check if there are any entries where ['PAID'] is NaN.
    If yes, drop entries with PAID = NaN
    
    NaN is not dropped if there is only one entry which also 
    happens to have PAID = NaN

    Parameters
    ----------
    df : pd.core.frame.DataFrame
        Should containg ['COMMON'] column which contains
        INCHI_KEY, UNIPROT_ID, Interaction Value,
        Reference [DOI or PATENT] data
        
        Should contain ['PAID'] column

    Returns
    -------
    nan_list : list
        list of index for which entries have
        PAID = NaN.

    """
    dupe_doi = list(set(df['COMMON']))
    nan_list = []
    for i in tqdm(dupe_doi):
        temp_df = df[df['COMMON'] == i]
        if len(temp_df) > 1:
            nan_indices = temp_df[temp_df['PAID'].isna()].index
            nan_list.extend(nan_indices)
            
    return nan_list

def mN_main(bdb_interact_path , bdb_final_path):
    # Find all BDB Interact files
    files = os.listdir(bdb_interact_path)
    try:
        files.remove('.gitkeep')
    except ValueError:
        pass
    
    # Iterate over each file name
    for file in tqdm(files):
        
        file_path = f'{bdb_interact_path}/{file}'
        final_file_path = f'{bdb_final_path}/{file}'
        
        # file = 'BDB_KI.tsv'
        
        if os.path.exists(final_file_path):
            print(f'{final_file_path} already exists. Skipping...')
            
        else:
            # Read the BDB Interaction file  
            df = pd.read_csv(file_path, sep = '\t')
            df['INTERACT'] = df['LIG_ID'] + '<>' + df['GENE_ID']
            
            
            # Separate the interactions based on if it has a duplicate entry or not
            # Separation is exclusive
            mask = df.duplicated(subset = ['INTERACT' , 'REF'], keep = False)
            df_c = df[mask]
            df_c.drop_duplicates(subset = ['INTERACT' , 'REF' , 'PATENT' , 'PAID' , 'DOI'],
                                 inplace = True)
            df_u = df[~mask]
            
            
            # Separate duplicate entry dataframe depeding on presence of DOI
            doi_y , doi_n = separate_doi(df_c)
            doi_y.reset_index(inplace = True, drop = True)
            
            
            # Separate duplicate entries with NO DOI depending on presence of PATENT
            pat_y , pat_n = separate_pat(doi_n)
            pat_y.reset_index(inplace = True , drop = True)
            
            
            # Separate duplicate entries with NO DOI & PATENT depending on
            # presence of PMID
            pmi_y , pmi_n = separate_pmid(pat_n)
            pmi_y.reset_index(inplace = True , drop = True)
            
            
            # Create new column containing the importan data
            doi_y['COMMON'] = doi_y['INTERACT'] + '||' + doi_y['REF'] + '||' + doi_y['DOI']
            pat_y['COMMON'] = pat_y['INTERACT'] + '||' + pat_y['REF'] + '||' + pat_y['PATENT']
            
            
            # Finding duplicate entries with common DOI or PATENT & marking index of
            # entries which does not have PAID
            doi_nan_list = check_nan_paid(doi_y)
            pat_nan_list = check_nan_paid(pat_y)
            
            
            # Remove these entries using index
            if doi_nan_list:
                doi_y.drop(doi_nan_list , inplace = True)
            else:
                pass
            
            if pat_nan_list:
                pat_y.drop(pat_nan_list , inplace = True)
            else:
                pass
            
            # Creating cleaned dataframe
            new_df = pd.concat([df_u , doi_y , pat_y])
            
            new_df.to_csv(final_file_path , sep = '\t',
                          index = False)
    
def mY_main(bdb_interact_path , bdb_final_path):
    # Find all BDB Interact files
    files = os.listdir(bdb_interact_path)
    try:
        files.remove('.gitkeep')
    except ValueError:
        pass
    
    # Iterate over each file name
    for file in tqdm(files):
        
        # file = 'BDB_KI.tsv'
        file_path = f'{bdb_interact_path}/{file}'
        final_file_path = f'{bdb_final_path}/{file}'
        
        if os.path.exists(final_file_path):
            print(f'{final_file_path} already exists. Skipping...')
            
        else:
    
            # Read the BDB Interaction file  
            df = pd.read_csv(file_path, sep = '\t')
            df['INTERACT'] = df['LIG_ID'] + '<>' + df['GENE_ID']
            
            
            # Separate the interactions based on if it has a duplicate entry or not
            # Separation is exclusive
            mask = df.duplicated(subset = ['INTERACT' , 'REF'], keep = False)
            df_c = df[mask]
            df_c.drop_duplicates(subset = ['INTERACT' , 'REF' , 'PATENT' , 'PAID' , 'DOI'],
                                 inplace = True)
            df_u = df[~mask]
            
            
            # Separate duplicate entry dataframe depeding on presence of DOI
            doi_y , doi_n = separate_doi(df_c)
            doi_y.reset_index(inplace = True, drop = True)
            
            
            # Separate duplicate entries with NO DOI depending on presence of PATENT
            pat_y , pat_n = separate_pat(doi_n)
            pat_y.reset_index(inplace = True , drop = True)
            
            
            # Separate duplicate entries with NO DOI & PATENT depending on
            # presence of PMID
            pmi_y , pmi_n = separate_pmid(pat_n)
            pmi_y.reset_index(inplace = True , drop = True)
            
            
            # Create new column containing the importan data
            doi_y['COMMON'] = doi_y['INTERACT'] + '||' + doi_y['REF'] + '||' + doi_y['DOI']
            pat_y['COMMON'] = pat_y['INTERACT'] + '||' + pat_y['REF'] + '||' + pat_y['PATENT']
            
            
            # Finding duplicate entries with common DOI or PATENT & marking index of
            # entries which does not have PAID
            doi_nan_list = check_nan_paid(doi_y)
            pat_nan_list = check_nan_paid(pat_y)
            
            
            # Remove these entries using index
            if doi_nan_list:
                doi_y.drop(doi_nan_list , inplace = True)
            else:
                pass
            
            if pat_nan_list:
                pat_y.drop(pat_nan_list , inplace = True)
            else:
                pass
            
            # Creating cleaned dataframe
            new_df = pd.concat([df_u , doi_y , pat_y])
            
            new_df.to_csv(final_file_path , sep = '\t',
                          index = False)     

def bdb_main6():
    
    
    mutN_int_path = 'processed/no_muts/mN_interactions'
    mutN_int_final_path = 'processed/no_muts/mN_clean_interactions'

    muyY_int_path = 'processed/with_muts/mY_interactions'
    mutY_int_final_path = 'processed/with_muts/mY_clean_interactions'
    
    
    if not os.path.exists(mutN_int_final_path):
        os.makedirs(mutN_int_final_path)
        print(f'Folder "{mutN_int_final_path}" created')
        
    if not os.path.exists(mutY_int_final_path):
        os.makedirs(mutY_int_final_path)
        print(f'Folder "{mutY_int_final_path }" created')
        
    
    
    print('Cleaning separated interactions with No mutations')
    mN_main(mutN_int_path , mutN_int_final_path)
    print('Done')
    
    print('Cleaning separated interactions with Yes mutations')
    mY_main(muyY_int_path , mutY_int_final_path)
    print('Done')
