#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:28:06 2024

@author: vka24
"""

import pandas as pd
import numpy as np
import ast
from tqdm import tqdm
import os

import warnings

# To suppress pandas warnings globally:
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


def clean_interact1(df : pd.core.frame.DataFrame):
    """
    Cleaning the Interaction Dataset
    
    1) Dropping interactions with no references
    2) Dropping duplicate interactions with same reference
        [Since it is unknown whether duplicate interactions with same
         reference have same assay (PAID) or not]
    3) Removing values for which reference is a broken link
    4) Creating column [REFERENCE_VALUE] which has original value
    5) Creating column [VALUE] which has int/float values. Value originally
        in the form of ">x" , "<x" & "x-y" is NaN here, but it is retained
        in the [REFERENCE_VALUE]

    Parameters
    ----------
    df : pd.core.frame.DataFrame
        Interaction Dataset

    Returns
    -------
    df : pd.core.frame.DataFrame
        Cleaned Interaction Dataset

    """
    ### CLEAN THE INTERACTION DATASET
    
    # Drop rows which does not have reference
    df.dropna(subset = ['REFERENCE'], inplace = True)
    
    
    # Drop rows which have duplicate interaction values & reference
    df.drop_duplicates(subset = ['INCHI_KEY' , 'ID' , 'PARAMETER' , 'VALUE' , 'UNIT' , 'REFERENCE'],
                       inplace = True)
    
    
    # Splitting dataset based on if reference is valid PMID
    df['REFERENCE_MASK'] = pd.to_numeric(df['REFERENCE'] , errors = 'coerce')
    
    # Valid PMID
    df_refY = df[~df['REFERENCE_MASK'].isnull()]
    df_refY.drop(['REFERENCE_MASK'], inplace = True, axis = 1)
    
    # Everything Else
    df_refN = df[df['REFERENCE_MASK'].isnull()]
    df_refN.drop(['REFERENCE_MASK'], inplace = True, axis = 1)
    
    # List of references for which links are broken
    broken_links = ['http://www.sciencedirect.com/science?_ob=ArticleURL&_udi=B6W84-42KD8D9-7&_coverDate=01%2F31%2F2001&_alid=93617739&_rdoc=1&_fmt=&_orig=search&_qd=1&_cdi=6644&_sort=d&view=c&_acct=C000050221&_version=1&_urlVersion=0&_userid=10&md5=5e5748d1ff8f5317fb6669f7ddc989f6',
                     'http://www.ncbi.nlm.nih.gov/entrez/query.fcgi?CMD=search&DB=pubmed',
                     'http://pubs.acs.org/cgi-bin/archive.cgi/jacsat/1998/120/i07/html/ja973325x.html',
                     'http://pdsp.med.unc.edu/pdsp.php',
                     'http://ac.els-cdn.com/S0223523414006928/1-s2.0-S0223523414006928-main.pdf?_tid=25a0770a-c81e-11e4-837d-00000aab0f26&acdnat=1426099433_3a4f5ff216f4fbe5be2ed79888bac358',
                     'sorry#http://sorry#']
    
    # List of references which does not give enough reference information
    # Not used here
    unknown_ref = ['PubChem BioAssay data set',
                   'DrugMatrix in vitro pharmacology data']
    
    # Removing interactions with broken links
    df_refN = df_refN[~df_refN['REFERENCE'].isin(broken_links)]
    
    # Removing imteractions with unknown reference
    #df_refN = df_refN[~df_refN['REFERENCE'].isin(unknown_ref)]
    
    
    # Combining the datasets together
    df = pd.concat([df_refY , df_refN])
    
    
    # Keeping non-int/float values as nan,
    # (>x & range x-y) values
    # but keep it as a reference in 'REFERENCE_VALUE
    df.rename(columns = {'VALUE' : 'REFERENCE_VALUE'},
              inplace = True)
    df.insert(3 , 'VALUE' , pd.to_numeric(df['REFERENCE_VALUE'] , errors = 'coerce'))
    
    
    # Clean the PARAMETER column
    
    df['PARAMETER'] = df['PARAMETER'].str.replace(" " , '_')
    
    # \ gives SyntaxError -> Replace It
    df['PARAMETER'] = df['PARAMETER'].str.replace("/" , ':')
    df['PARAMETER'] = df['PARAMETER'].str.replace("\\" , ':')
    
    return df


### ADD THE LIGAND TO THE DATASET
def clean_interact2(df : pd.core.frame.DataFrame,
                 gene_id : pd.core.frame.DataFrame,
                 lig_id : pd.core.frame.DataFrame):
    """
    1)Add unique Gene & Ligand ID to interaction dataset.
    2) Some interactions are the same [with same reference], but was kept separate due to 
        human errors in the values.
        Cleaned as much as possible.

    Parameters
    ----------
    df : pd.core.frame.DataFrame
        Interaction Dataset
        
    gene_id : pd.core.frame.DataFrame
        Gene Info Dataset
        
    lig_id : pd.core.frame.DataFrame
        Ligand Info Dataset

    Returns
    -------
    None.

    """
    
    # Setting index
    gene_id.set_index('ID', inplace = True)
    lig_id.set_index('INCHI_KEY' , inplace = True)
    
    # Map the IDs to UNIPROT_ID & INCHI_KEY
    df.insert(1 , 'GENE_ID' , df['ID'].map(gene_id['GENE_ID']))
    df.insert(0 , 'LIG_ID' , df['INCHI_KEY'].map(lig_id['LIG_ID']))
    
    # Move ID & INCHI_KEY to the back
    col_list = list(df.columns)
    col_list.append(col_list.pop(1))
    col_list.append(col_list.pop(2))
    
    df = df[col_list]
    
    # Some rows are same but are not considered duplicate even though
    # it has the same reference. This is because the values different by a very
    # small amount (probably human error) & sometimes it is differentiated
    # because one row has trailing zeroes
    
    # These are duplicates and should be fixed here
    # interact1 10        ref1
    # interact1 10.000000 ref1
    
    # These are also duplicates. Most of the times, it appears in different
    # datasources like Chembl/BindingDB
    # interact1 10        ref1
    # interact1 10.000001 ref1
    # But we will lose some values if it is actually not duplicate values
    
    # Separating values based on if it is null or not
    df_null = df[df['VALUE'].isnull()]
    df_fill = df[~df['VALUE'].isnull()]
    
    # Separating the values based on if >100
    df_above100 = df_fill[df_fill['VALUE'] > 100]
    
    # Separating value based on 1 < value <= 100
    df_above1 = df_fill[df_fill['VALUE'] > 1]
    df_above1 = df_above1[df_above1['VALUE'] <= 100]
    
    # Separating value based on value <= 1
    df_below1 = df_fill[df_fill['VALUE'] <= 1]
    
    # Applying rounding for values greater than 100
    df_above100['VALUE'] = np.floor(df_above100['VALUE'])
    
    # Rounding to 1 digit for 1 < value <= 100
    df_above1['VALUE'] = df_above1['VALUE'].round(1)
    
    # Rounding to 3 digits for value <= 1
    df_below1['VALUE'] = df_below1['VALUE'].round(3)
    
    # Dropping based on value will work somewhat now. However, some errors
    # definetly passed this.
    # For example, values which are actually not the same  with same reference could have been
    # rounded to the same no, and thus removed
    
    # Another issue is for some values, the round might not have been enough
    df = pd.concat([df_null , df_above100 , df_above1 , df_below1])
    df.drop_duplicates(subset = ['GENE_ID' , 'LIG_ID' , 'PARAMETER' , 'VALUE' , 'REFERENCE'],
                       inplace = True)
    
    return df


def glass_main4():
    
    int_loc = 'processed/old/GLASS_Interaction.tsv'
    
    clean_int_loc = 'processed/GLASS_Interaction_Fixed.tsv'
    clean_lig_id_loc = 'processed/GLASS_Ligand_ID.tsv'
    clean_tar_id_loc = 'processed/GLASS_Target_ID.tsv'
    
    
    if os.path.exists(clean_int_loc):
        print(f'{clean_int_loc} already exists. Skipping...')
        
    else:
        # Read interaction dataset
        df = pd.read_csv(int_loc,
                         sep = '\t')
        # Open the ID files
        gene_id = pd.read_csv(clean_tar_id_loc, sep = '\t')
        lig_id = pd.read_csv(clean_lig_id_loc , sep = '\t')
        
        print('Cleaning Interaction dataset')
        df = clean_interact1(df)
        print('Done')
        
        print('Adding Tar & Lig IDs and cleaning based on decimals')
        df = clean_interact2(df, gene_id, lig_id)
        print('Done')
        
        
        df.to_csv(clean_int_loc,
                  sep = '\t',
                  index = False)
