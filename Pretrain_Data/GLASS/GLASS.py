# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 14:09:51 2024

@author: HP
"""

import pandas as pd
import numpy as np

import sys
import os
from typing import Union

import warnings

# To suppress pandas warnings globally:
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


def target_data(target_df : pd.core.frame.DataFrame):
    """
    Clean GLASS target dataset

    Parameters
    ----------
    target_df : pd.core.frame.DataFrame

    Returns
    -------
    ligand_df : pd.core.frame.DataFrame

    """
    
    # Renaming columns
    target_df.rename(columns = {'GPCR Name' : 'NAME',
                                'UniProt ID' : 'ID',
                                'Gene Name' : 'GENE',
                                'Species' : 'ORGANISM',
                                'FASTA Sequence' : 'FASTA'},
                     inplace = True)
    
    # Database where Target is found
    target_df['GLASS'] = 1
    
    def remove_parenthesis_organism(text : str):
        """
        Small function to remove the source organism name in parenthesis
        in the ORGANISM column

        Parameters
        ----------
        text : str

        Returns
        -------
        text : str

        """
        return text.split('(' , 1 )[0]
    
    # Source organism column converted : "Homo sapiens (Humans)" -> "Homo sapiens"
    target_df['ORGANISM'] = target_df['ORGANISM'].apply(remove_parenthesis_organism)
    
    # Rearranging Columns (For Sanity)
    rearranged_cols = ['ID' , 'GENE' , 'NAME' , 'ORGANISM' , 'FASTA']
    
    target_df = target_df[rearranged_cols]
    
    return target_df

def ligand_data(ligand_df : pd.core.frame.DataFrame):
    """
    Clean GLASS ligand dataset

    Parameters
    ----------
    ligand_df : pd.core.frame.DataFrame

    Returns
    -------
    ligand_df : pd.core.frame.DataFrame

    """
    
    # Renaming the Columns
    ligand_df.rename(columns = {'Ligand Name' : 'GLASS_NAME',
                            'CID' : 'PCID',
                            'Molecular Formula' : 'MOL_FORMULA',
                            'Molecular Weight' : 'MOL_WT',
                            'IUPAC Name' : 'IUPAC',
                            'Canonical SMILES' : 'GLASS_CANON_SMILES',
                            'Isomeric SMILES' : 'GLASS_ISO_SMILES',
                            'InChI Std. ID' : 'INCHI',
                            'InChI Key' : 'INCHI_KEY',
                            'XlogP' : 'XLOGP',
                            'Hydrogen Bond Donors' : 'HYD_BOND_DONOR',
                            'Hydrogen Bond Acceptors' : 'HYD_BOND_ACCEPTOR'},
                 inplace = True)

    # Rearranging columns (For Sanity)  
    rearranged_cols = ['INCHI_KEY' , 'INCHI' , 'GLASS_CANON_SMILES',
                       'GLASS_ISO_SMILES' , 'GLASS_NAME' , 'PCID' , 
                       'MOL_FORMULA' , 'MOL_WT' , 'XLOGP' , 'HYD_BOND_DONOR',
                       'HYD_BOND_ACCEPTOR']
    
    ligand_df = ligand_df[rearranged_cols]   
    
    return ligand_df

def interaction_data(interaction_df : pd.core.frame.DataFrame):
    """
    Clean GLASS Interaction dataset

    Parameters
    ----------
    interaction_df : pd.core.frame.DataFrame

    Returns
    -------
    interaction_df : pd.core.frame.DataFrame

    """
    
    interaction_df.rename(columns = {"UniProt ID" : "ID",
                                      "InChI Key" : "INCHI_KEY",
                                      "Database Source" : "DATASOURCE",
                                      "Database Target ID" : "DATASOURCE_TARGET_ID",
                                      "Database Ligand ID" : "DATASOURCE_LIGAND_ID",
                                      "Reference" : "REFERENCE",
                                      "Unit" : "UNIT",
                                      'Parameter' : 'PARAMETER',
                                      'Value' : 'VALUE'},
                 inplace = True)
    
    # Value column has spaces
    interaction_df['VALUE'] = interaction_df['VALUE'].str.replace(" " , "")
    
    # Rearranging columns (For Sanity)
    rearranged_cols = ['INCHI_KEY' , 'ID' , 'PARAMETER' , 'VALUE',
                       'UNIT' , 'REFERENCE' , 'DATASOURCE',
                       'DATASOURCE_TARGET_ID', 'DATASOURCE_LIGAND_ID']
    
    interaction_df = interaction_df[rearranged_cols]
    
    return interaction_df

    
def glass_main1(data_target_loc , data_ligand_loc , data_interact_loc):
    
    # Path to final datasets
    processed_path = 'processed'
    old_path = 'processed/old' # Where the cleaned but not fixed datasets are located
    processed_target_loc = 'processed/old/GLASS_Target.tsv'
    processed_ligand_loc = 'processed/old/GLASS_Ligand.tsv'
    processed_interact_loc = 'processed/old/GLASS_Interaction.tsv'
    
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
        print(f'Folder "{processed_path}" created')
        
    if not os.path.exists(old_path):
        os.makedirs(old_path)
        print(f'Folder "{old_path}" created')
    
    
    # TARGET data
    if os.path.exists(processed_target_loc):
        print(f'{processed_target_loc} already exists. Skipping...')
    
    else:
        print('Cleaning Target Data')
        target = pd.read_csv(data_target_loc, sep = '\t')
        target = target_data(target) # Clean target data
        target.drop_duplicates(subset = ['ID'] , inplace = True) # No change
        
        # Write to TARGET Dataset
        target.to_csv(processed_target_loc,
                      sep = '\t',
                      index = False)
        del target
        print('Done')
        
    
    # LIGAND Data
    if os.path.exists(processed_ligand_loc):
        print(f'{processed_ligand_loc} already exists. Skipping...')
        
    else:  
        print('Cleaning Ligand Data')
        ligand = pd.read_csv(data_ligand_loc, sep = '\t')
        ligand = ligand_data(ligand)
        ligand.drop_duplicates(subset = ['INCHI_KEY'] , inplace = True) # No change
        
        # Write to LIGAND Dataset
        ligand.to_csv(processed_ligand_loc,
                      sep = '\t',
                      index = False)
        del ligand
        print('Done')
    
    
    # INTERACTION dataset
    if os.path.exists(processed_interact_loc):
        print(f'{processed_interact_loc} already exists. Skipping...')
    
    else:
        print('Cleaning Interaction data')
        interact = pd.read_csv(data_interact_loc, sep = '\t')
        interact = interaction_data(interact)

        # Write to INTERACTION Dataset
        interact.to_csv(processed_interact_loc,
                        sep = '\t',
                        index = False)
        del interact
        print('Done')

                          







