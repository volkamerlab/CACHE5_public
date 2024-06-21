#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 18:39:00 2024

@author: vka24
"""

import pandas as pd
import requests
from tqdm import tqdm

import os
import warnings

# To suppress pandas warnings globally:
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


def bdb_main3():
    """
    

    Parameters
    ----------
    lig_loc : str
        Cleaned Liganad Dataset Loc
        
    dupe_pcid_loc : str
        Ligand with duplicate PCID Dataset Loc
        
    mutN_lig_loc : str
        Copy of Cleaned Ligand dataset for No Mutations Dataset Loc
    mutN_lig_id_loc : str
        
    mutY_lig_loc : str
        Copy of Cleaned Ligand dataset for Yes Mutations Dataset Loc
        
    mutY_lig_id_loc : str
        

    Returns
    -------
    None.

    """
    lig_loc = "processed/old/BDB_Ligand.tsv"
    
    dupe_pcid_loc = 'processed/old/BDB_Ligand_DupePCID.tsv'

    mutN_lig_loc = 'processed/no_muts/BDB_mN_Ligand_Clean.tsv'
    mutN_lig_id_loc = 'processed/no_muts/BDB_mN_Ligand_IDs.tsv'

    mutY_lig_loc = 'processed/with_muts/BDB_mY_Ligand_Clean.tsv'
    mutY_lig_id_loc = 'processed/with_muts/BDB_mY_Ligand_IDs.tsv'
    
    # Read file
    df = pd.read_csv(lig_loc, sep = '\t')
    
    print('Fixing Ligand Data')
    # Small clean
    df.drop_duplicates(subset = ['INCHI_KEY'], inplace = True)
    df['PCID'] = df['PCID'].astype(int)
    df['PCID'] = df['PCID'].astype(str)
    
    # There are different ligands which have same PCID
    # Remove them
    mask = df.duplicated(subset = ['PCID'], keep = False)
    df_uniq = df[~mask]
    df_dupe = df[mask]
    df_dupe.to_csv(dupe_pcid_loc,
                   sep = '\t',
                   index = False)
    print('Done')
    
    
    if os.path.exists(mutN_lig_loc) and os.path.exists(mutN_lig_id_loc) and os.path.exists(mutY_lig_loc) and os.path.exists(mutY_lig_id_loc):
        print(f'{mutN_lig_loc} , {mutN_lig_id_loc} , {mutY_lig_loc} & {mutY_lig_id_loc} already exists. Skipping...')
        
    else:
        print('Creating ligand IDs')
        df_uniq['TEMP'] = range(1 , len(df_uniq)+1)
        df_uniq.insert(0 , 'LIG_ID' , 'L' + df_uniq['TEMP'].astype(str))
        df_uniq.drop('TEMP' , axis = 1 , inplace = True)
        
        lig_id = df_uniq[['LIG_ID' , 'INCHI_KEY']]
        lig_id.to_csv(mutN_lig_id_loc, sep = '\t', index = False)
        lig_id.to_csv(mutY_lig_id_loc, sep = '\t', index = False)
        
        
        df_uniq.to_csv(mutN_lig_loc,
                  sep = '\t',
                  index = False)
        
        df_uniq.to_csv(mutY_lig_loc,
                  sep = '\t',
                  index = False)
        
        print('Done')



def pcid2properties(pcid):
    """
    Call molecule properties from pubchem for 
    the Pubchem CID
    
    Properties
    ----------
    InChIKey
    InChI
    Canonical SMILES
    Isomeric SMILES
    Molecular Formula
    Molecular Weight
    XLogP

    Parameters
    ----------
    pcid : str
        Pubchem Compound Identifier
        Should either be 'x'
        or 'x,y,z' if multiple [len 10 good]
        

    Returns
    -------
    table : dict/None

    """
    
    properties = 'InChiKey,InChi,CanonicalSMILES,IsomericSMILES,MolecularFormula,MolecularWeight,XLogP'
    
    url = f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{pcid}/property/{properties}/JSON'
    
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        table = data['PropertyTable']['Properties']
        
        table_df = pd.DataFrame(table)
        
        return table_df
    
    else:
        return None
    
# Using PCID to get the molecular properties
# mol_df = pd.DataFrame()
# pcid_list = list(set(df_uniq['PCID']))

# for i in tqdm(range(0 , len(pcid_list) + 1 , 300)):
#     pcid_i = ','.join(pcid_list[i : i+300])
    
#     temp_df = pcid2properties(pcid_i)
#     mol_df = pd.concat([mol_df , temp_df])
    
#     if type(temp_df) != pd.core.frame.DataFrame:
#         break