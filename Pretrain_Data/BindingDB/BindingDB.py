# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 18:07:49 2024

@author: HP
"""


import numpy as np
import pandas as pd

import os 
import sys
import argparse

from tqdm import tqdm

import warnings

# To suppress pandas warnings globally:
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
# sys.path.append('scripts')


def rename_cols(df: pd.core.frame.DataFrame):
    """
    Rename Column Names of BindingDB DataFrame

    Parameters
    ----------
    df : pd.core.frame.DataFrame

    Returns
    -------
    None.

    """

    df.rename(columns=  # Ligand Columns
              {'Ligand SMILES': 'BDB_SMILES',
                  'Ligand InChI': 'INCHI',
                  'Ligand InChI Key': "INCHI_KEY",
               'BindingDB Ligand Name': "BDB_NAME",
               'Ligand HET ID in PDB': 'PDB_HET_ID',
               'PubChem CID': "PCID",
               'PubChem SID': "PSID",
               'ChEBI ID of Ligand': 'CHEBI',
               'ChEMBL ID of Ligand': 'CHEMBL',
               'DrugBank ID of Ligand': 'DRUGB',
               'IUPHAR_GRAC ID of Ligand': 'IUPHAR_GRAC',
               'KEGG ID of Ligand': 'KEGG',
               'ZINC ID of Ligand': 'ZINC',
               "BindingDB MonomerID": 'BDBM_ID',

               # Target Columns
               'Target Name': 'NAME',
               'Target Source Organism According to Curator or DataSource': "SOURCE_ORGANISM",
               'Number of Protein Chains in Target (>1 implies a multichain complex)': "NO_PROT_CHAIN",
               'BindingDB Target Chain Sequence': 'BDB_CHAIN_SEQ',
               'PDB ID(s) of Target Chain': "PDB_ID",
               'UniProt (SwissProt) Recommended Name of Target Chain': "SWISSPROT_NAME",
               'UniProt (SwissProt) Entry Name of Target Chain': "SWISSPROT_ENTRY",
               'UniProt (SwissProt) Primary ID of Target Chain': "SWISSPROT_ID",
               'UniProt (SwissProt) Secondary ID(s) of Target Chain': "SWISSPROT_SECONDARY_ID",
               'UniProt (SwissProt) Alternative ID(s) of Target Chain': "SWISSPROT_ALT_ID",
               'UniProt (TrEMBL) Submitted Name of Target Chain': "TREMBL_NAME",
               'UniProt (TrEMBL) Entry Name of Target Chain': "TREMBL_ENTRY",
               'UniProt (TrEMBL) Primary ID of Target Chain': "TREMBL_PRIMARY_ID",
               'UniProt (TrEMBL) Secondary ID(s) of Target Chain': "TREMBL_SECONDARY_ID",
               'UniProt (TrEMBL) Alternative ID(s) of Target Chain': "TREMBL_ALT_ID",

               # Interaction Columns
               'BindingDB Reactant_set_id': "BDB_REACTANT_ID",
               'Ligand InChI Key': "INCHI_KEY",
               'Ki (nM)': "KI",
               'IC50 (nM)': "IC50",
               'Kd (nM)': "KD",
               'EC50 (nM)': 'EC50',
               'kon (M-1-s-1)': "KON",
               'koff (s-1)': "KOFF",
               'pH': "PH",
               'Temp (C)': "TEMP",
               'Curation/DataSource': "DATASOURCE",
               'Article DOI': "DOI",
               'BindingDB Entry DOI': "BDB_DOI",
               'PubChem AID': "PAID",
               'Patent Number': "PATENT",
               'Authors': 'AUTHOR',
               'Institution': "INSTITUTE",
               'PDB ID(s) for Ligand-Target Complex': "COMPLEX_PDB_ID"},
              inplace=True)


def fix_interaction(chunk: pd.core.frame.DataFrame):
    """
    Fix irregularities in columns of the interaction chunk

    Parameters
    ----------
    chunk : pd.core.frame.DataFrame
        A chunk of BindingDB data with 100,000 rows

    Returns
    -------
    None.

    """

    # Temp values is of form "xx.xx C"
    try:
        chunk['TEMP'] = chunk['TEMP'].str.replace(" C", "")
    except AttributeError:
        pass

    # Coefficient value columns has spaces(" ")
    # Some chunks might have columns without these spaces
    # Therefore, check attribute
    try:
        chunk['KI'] = chunk['KI'].str.replace(" ", "")
    except AttributeError:
        pass

    try:
        chunk['IC50'] = chunk['IC50'].str.replace(" ", "")
    except AttributeError:
        pass

    try:
        chunk['KD'] = chunk['KD'].str.replace(" ", "")
    except AttributeError:
        pass

    try:
        chunk['EC50'] = chunk['EC50'].str.replace(" ", "")
    except AttributeError:
        pass

    try:
        chunk['KON'] = chunk['KON'].str.replace(" ", "")
    except AttributeError:
        pass

    try:
        chunk['KOFF'] = chunk['KOFF'].str.replace(" ", "")
    except AttributeError:
        pass


def create_gene_name(name: str, uniprot_id: str):
    """
    Create a unique gene ID based on UNIPROT ID & 
    Mutation/Loc of Interaction

    Parameters
    ----------
    name : str
        Gene Name [BDB adds location/mutation data in gene name]
    uniprot_id : str
        UNIPROT ID.

    Returns
    -------
    new_id : str
        "UNIPROT_ID::MUTATION" / "UNIPROT_ID"

    """

    split_name = name.split('[', 1)

    if len(split_name) == 2:
        try:
            mut = '[' + split_name[1]
            new_id = uniprot_id + "::" + mut

        except TypeError:
            print(mut)
            print(name)
            print(uniprot_id)

        return new_id
    else:
        new_id = uniprot_id
        return new_id


def ligand_chunk(df: pd.core.frame.DataFrame):
    """
    Separate Ligand information from overall dataset

    Parameters
    ----------
    df : pd.core.frame.DataFrame

    Returns
    -------
    lig_chunk : pd.core.frame.DataFrame

    """

    lig_chunk = df[['BDB_SMILES', 'INCHI', 'INCHI_KEY', 'BDB_NAME',
                    'PDB_HET_ID', 'PCID', 'PSID', 'CHEBI', 'CHEMBL',
                    'DRUGB', 'IUPHAR_GRAC', 'KEGG', 'ZINC', 'BDBM_ID']]

    # Rearranging columns (For Sanity)
    lig_chunk = lig_chunk[['INCHI_KEY', 'INCHI', 'BDB_SMILES',
                           'BDB_NAME', 'BDBM_ID', 'PCID', 'PSID',
                           'CHEBI', 'CHEMBL', 'DRUGB', 'IUPHAR_GRAC',
                           'KEGG', 'ZINC']]

    return lig_chunk


def target_chunk(df: pd.core.frame.DataFrame):
    """
    Separate Target information from overall dataset

    Parameters
    ----------
    df : pd.core.frame.DataFrame

    Returns
    -------
    tar_chunk : pd.core.frame.DataFrame

    """

    tar_chunk = df[['ID', 'NAME', 'SOURCE_ORGANISM', 'NO_PROT_CHAIN',
                    'BDB_CHAIN_SEQ', 'PDB_ID', 'SWISSPROT_NAME',
                    'SWISSPROT_ENTRY', 'SWISSPROT_ID',
                    'SWISSPROT_SECONDARY_ID', 'SWISSPROT_ALT_ID',
                    'TREMBL_NAME', 'TREMBL_ENTRY', 'TREMBL_PRIMARY_ID',
                    'TREMBL_SECONDARY_ID', 'TREMBL_ALT_ID',
                    'UNIPROT_ID']]


    # Rearranging columns (For Sanity)
    tar_chunk = tar_chunk[['ID', 'UNIPROT_ID', 'NAME', 'SOURCE_ORGANISM',
                           'PDB_ID', 'SWISSPROT_NAME', 'SWISSPROT_ENTRY',
                           'SWISSPROT_ID', 'SWISSPROT_SECONDARY_ID',
                           'SWISSPROT_ALT_ID', 'TREMBL_NAME',
                           'TREMBL_ENTRY', 'TREMBL_PRIMARY_ID',
                           'TREMBL_SECONDARY_ID', 'TREMBL_ALT_ID', 'NO_PROT_CHAIN',
                           'BDB_CHAIN_SEQ']]

    return tar_chunk


def interact_chunk(df: pd.core.frame.DataFrame):
    """
    Separate Interaction information from overall dataset

    Parameters
    ----------
    df : pd.core.frame.DataFrame

    Returns
    -------
    int_chunk : pd.core.frame.DataFrame

    """

    int_chunk = df[['BDB_REACTANT_ID', 'INCHI_KEY', 'UNIPROT_ID', 'ID',
                    'KI', 'IC50', 'KD', 'EC50', 'KON', 'KOFF', 'PH',
                    'TEMP', 'DATASOURCE', 'DOI', 'BDB_DOI', 'PAID',
                    'PATENT', 'AUTHOR', 'INSTITUTE', 'COMPLEX_PDB_ID',
                    'PMID']]


    # Rearranging columns (For Sanity)
    int_chunk = int_chunk[['INCHI_KEY', 'ID', 'UNIPROT_ID', 'KI', 'IC50',
                           'KD', 'EC50', 'KON', 'KOFF', 'PH',
                           'TEMP', 'BDB_REACTANT_ID',
                           'COMPLEX_PDB_ID', 'DATASOURCE',
                           'DOI', 'BDB_DOI', 'PMID', 'PAID',
                           'PATENT', 'AUTHOR', 'INSTITUTE']]

    return int_chunk


def clean_BDB(df_loc: str, clean_chunk_loc: str, nan_chunk_loc: str,
              chunk_size : int , clean_temp_loc : str , nan_temp_loc : str):
    """
    Separates the dataset into CLEAN & NAN Datasets. Separated based on 
    UNIPROT ID, BDB SMILES, INCHI, INCHI KEY & PCID NAN Values.

    Parameters
    ----------
    df_loc : str
        Location of BDB Dataset

    clean_chunk_loc : str
        Location to store CLEAN BDB Dataset

    nan_chunk_loc : str
        Location to store NAN BDB Dataset
        
    chunk-size : int
        Size of chunks
        
    clean_temp_loc : str
        Location of temp clean data file
        
    nan_temp_file : str
        Location of temp nan data file

    Returns
    -------
    None

    """

    bdb = pd.read_csv(df_loc,
                      sep='\t',
                      # nrows = 0,
                      chunksize=chunk_size,
                      usecols=range(0, 50))

    initial_chunk = True
    print('Starting Dataset Cleaning')
    for chunk in tqdm(bdb):

        # Renaming column names
        rename_cols(chunk)

        # Merging SWISSPROT & TREMBL IDs
        chunk['UNIPROT_ID'] = chunk['SWISSPROT_ID'].fillna(
            chunk['TREMBL_PRIMARY_ID'])

        chunk.drop(columns=['Link to Ligand in BindingDB',
                            'Link to Target in BindingDB',
                            'Link to Ligand-Target Pair in BindingDB'],
                   inplace=True,
                   axis=1)

        # Fixing small problems with interaction value columns
        fix_interaction(chunk)

        # Separating the rows depending on NaN values in the columns
        clean_chunk = chunk.dropna(subset=['UNIPROT_ID', 'BDB_SMILES', 'INCHI',
                                           'INCHI_KEY', 'PCID'])

        nan_chunk = chunk[~chunk.index.isin(clean_chunk.index)]

        # Combining UNIPROT ID & Mutation/Location Data
        clean_chunk['ID'] = clean_chunk.apply(lambda row: create_gene_name(row['NAME'],
                                                                           row['UNIPROT_ID']),
                                              axis=1)

        # Creating processed BDB Files
        if initial_chunk:
            clean_chunk.to_csv(clean_temp_loc, sep='\t', index=False)

            nan_chunk.to_csv(nan_temp_loc, sep='\t', index=False)

            initial_chunk = False

        # Appendin the later chunks to BDB file
        else:
            clean_chunk.to_csv(clean_temp_loc,
                               sep='\t',
                               index=False,
                               header=False,
                               mode='a')

            nan_chunk.to_csv(nan_temp_loc,
                             sep='\t',
                             index=False,
                             header=False,
                             mode='a')
            
    os.rename(clean_temp_loc , clean_chunk_loc)
    os.rename(nan_temp_loc , nan_chunk_loc)
    print('Done')


def separate_BDB(clean_chunk_loc: str, tar_chunk_loc: str,
                 lig_chunk_loc: str, int_chunk_loc: str,
                 chunk_size : int , tar_temp_loc : str , lig_temp_loc : str,
                 int_temp_loc : str):
    """
    Extracts & separates the CLEAN BDB Dataset into Target, Ligand
    & Interaction Dataset

    Parameters
    ----------
    clean_chunk_loc : str
        Location of CLEAN BDB Dataset

    tar_chunk_loc : str
        Location to store Target BDB Data

    lig_chunk_loc : str
        Location to store Ligand BDB Data

    int_chunk_loc : str
        Location to store Interaction BDB Data
        
    chunk_size : int
        Size of chunks
        
    tar_temp_loc : str
        Location of temp tar data file
        
    lig_temp_loc : str
        Location of temp lig data file
        
    int_temp_loc : str
        Location of temp int data file

    Returns
    -------
    None.

    """

    bdb = pd.read_csv(clean_chunk_loc,
                      sep='\t',
                      # nrows = 0,
                      chunksize=chunk_size)

    initial_chunk = True
    print('Starting Dataset separation')
    for chunk in tqdm(bdb):

        # Extract Target, Ligand & Interaction data into different chunks
        tar_chunk = target_chunk(chunk)
        lig_chunk = ligand_chunk(chunk)
        int_chunk = interact_chunk(chunk)

        # Create a TSV file for each chunk
        if initial_chunk:
            tar_chunk.to_csv(tar_temp_loc, sep='\t', index=False)

            lig_chunk.to_csv(lig_temp_loc, sep='\t', index=False)

            int_chunk.to_csv(int_temp_loc, sep='\t', index=False)

            initial_chunk = False

        # Append following chunks to the proper TSV
        else:
            tar_chunk.to_csv(tar_temp_loc,
                             sep='\t',
                             index=False,
                             header=False,
                             mode='a')

            lig_chunk.to_csv(lig_temp_loc,
                             sep='\t',
                             index=False,
                             header=False,
                             mode='a')

            int_chunk.to_csv(int_temp_loc,
                             sep='\t',
                             index=False,
                             header=False,
                             mode='a')
            
    os.rename(tar_temp_loc , tar_chunk_loc)
    os.rename(lig_temp_loc , lig_chunk_loc)
    os.rename(int_temp_loc , int_chunk_loc)
            
    print("Done")


def obsolete_gene_ids(tar_df: pd.core.frame.DataFrame):
    """
    Remove IDs from Gene Dataset which are obsolete in UniProt

    Parameters
    ----------
    tar_df : pd.core.frame.DataFrame
        Target Dataframe

    Returns
    -------
    tar_df : pd.core.frame.DataFrame

    obs_df : pd.core.frame.DataFrame

    """

    # Finding Genes not having SWISSPROT NAME/ID
    temp = tar_df[tar_df['SWISSPROT_NAME'].isnull()]

    # Finding Genes not having TREMBL NAME. These have TREMBL IDs
    # but these IDs are obsolete in UniProt
    obs_df = temp[temp['TREMBL_NAME'].isnull()]
    obs_id = list(obs_df['ID'])

    # Removing these IDs from the Target Dataset
    tar_df = tar_df[~tar_df['ID'].isin(obs_id)]

    return (tar_df, obs_df)


def bdb_main1(df_loc , chunk_size = 500000):
    
    clean_chunk_loc = 'processed/old/BDB_CLEAN.tsv'
    clean_temp_loc = 'processed/old/BDB_CLEAN_temp.tsv'
    
    nan_chunk_loc = 'processed/old/BDB_NAN.tsv'
    nan_temp_loc = 'processed/old/BDB_NAN_temp.tsv'
    
    tar_chunk_loc = "processed/old/BDB_Target.tsv"
    tar_temp_loc = 'processed/old/BDB_Target_temp.tsv'
    
    lig_chunk_loc = "processed/old/BDB_Ligand.tsv"
    lig_temp_loc = 'processed/old/BDB_Ligand_temp.tsv'
    
    int_chunk_loc = "processed/old/BDB_Interaction.tsv"
    int_temp_loc = 'processed/old/BDB_Interaction_temp.tsv'
    
    obs_loc = 'processed/old/BDB_Target_Obs.tsv'

    
    if not os.path.exists("processed"):
        os.makedirs('processed')
        print('Folder "processed" created')
        
    if not os.path.exists("processed/old"):
        os.makedirs('processed/old')
        print('Folder "processed/old" created')
        
    
    if os.path.exists(clean_chunk_loc) and os.path.exists(nan_chunk_loc):
        print(f'{clean_chunk_loc} and {nan_chunk_loc} already present. Skipping...')
        
    else:
        clean_BDB(df_loc, clean_chunk_loc, nan_chunk_loc , chunk_size,
                  clean_temp_loc , nan_temp_loc)

    
    if os.path.exists(tar_chunk_loc) and os.path.exists(int_chunk_loc) and os.path.exists(lig_chunk_loc):
        print(f"{int_chunk_loc} , {tar_chunk_loc} & {lig_chunk_loc} already exists. Skipping...")
        
    else:
        separate_BDB(clean_chunk_loc, tar_chunk_loc, lig_chunk_loc, int_chunk_loc,
                     chunk_size , tar_temp_loc , lig_temp_loc , int_temp_loc)

    # Dropping duplicates from whole Target Datasets
    tar_df = pd.read_csv(tar_chunk_loc, sep='\t')
    tar_df.drop_duplicates(subset=['ID'], inplace=True)

    # Separating Genes by Proper & Obsolete UniProt IDs
    tar_df, obs_df = obsolete_gene_ids(tar_df)

    # Writing TSV File
    tar_df.to_csv(tar_chunk_loc, sep='\t', index=False)
    
    
    obs_df.to_csv(obs_loc, sep='\t', index=False)

    # Removing interactions with Obsolete IDs
    int_df = pd.read_csv(int_chunk_loc, sep='\t')
    int_df = int_df[~int_df['ID'].isin(list(obs_df['ID']))]
    

    int_df.to_csv(int_chunk_loc, sep='\t', index=False)

    # Dropping duplicates from Ligand Datasets
    lig_df = pd.read_csv(lig_chunk_loc, sep='\t')
    lig_df.drop_duplicates(subset=['INCHI_KEY'], inplace=True)
    lig_df.to_csv(lig_chunk_loc, sep='\t', index=False)
