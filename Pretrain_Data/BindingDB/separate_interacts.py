# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 19:44:49 2024

@author: HP
"""

import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm
import os

import warnings

# To suppress pandas warnings globally:
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


def bdb_main5():
    
    mutN_int_loc = 'processed/no_muts/BDB_mN_Interaction_Clean.tsv'
    mutY_int_loc = 'processed/with_muts/BDB_mY_Interaction_Clean.tsv'
    
    coeff = ['IC50' , 'KD' , 'EC50' , 'KON' , 'KOFF' , 'KI']
    
    mutN_sep_int_loc = "processed/no_muts/mN_interactions/BDB_mN_{}.tsv"
    mutY_sep_int_loc = "processed/with_muts/mY_interactions/BDB_mY_{}.tsv"
    
    if not os.path.exists('processed/no_muts/mN_interactions'):
        os.makedirs('processed/no_muts/mN_interactions')
        print('Folder "processed/no_muts/mN_interactions" created')
        
    if not os.path.exists('processed/with_muts/mY_interactions'):
        os.makedirs('processed/with_muts/mY_interactions')
        print('Folder "processed/with_muts/mY_interactions" created')
    
    
   
    # Without Mutations   
    bdb = pd.read_csv(mutN_int_loc, sep = '\t')
    
    print('Separating Interaction dataset with No mutations based on interaction coefficients')
    for i in tqdm(coeff):
        
        write2csv = mutN_sep_int_loc.format(i)
        
        if os.path.exists(write2csv):
            print(f'{write2csv} already exists. Skipping...')
            
        else: 
            # Find rows which has the coefficient values
            bdb_coeff = bdb[bdb[i].notna()]
            
            # Remove unneccessary columns
            unnec_coeff = coeff[:]
            unnec_coeff.remove(i)
            bdb_coeff.drop(columns = unnec_coeff,
                           inplace = True)
            
            # Create a copy of the coefficent value in new column
            # as a reference to unmodifed values
            bdb_coeff['REF'] = bdb_coeff[i]
            
            # Modify erronuous values in the interaction columns
            # Replace '<i' , '>i' & .NP,' values with np.nan
            bdb_coeff[i] = bdb_coeff[i].replace(['^>|^<|NV,'] , np.nan,
                                               regex = True)
            
            
            
            bdb_coeff.to_csv(write2csv,
                          sep = '\t',
                          index = False)
    print('Done')
        
        
    # With Mutations  
    bdb = pd.read_csv(mutY_int_loc , sep = '\t')
    
    print('Separating Interaction dataset with Yes mutations based on interaction coefficients')
    for i in tqdm(coeff):
        
        write2csv = mutY_sep_int_loc.format(i)
        
        if os.path.exists(write2csv):
            print(f'{write2csv} already exists. Skipping...')
            
        else:  
            # Find rows which has the coefficient values
            bdb_coeff = bdb[bdb[i].notna()]
            
            # Remove unneccessary columns
            unnec_coeff = coeff[:]
            unnec_coeff.remove(i)
            bdb_coeff.drop(columns = unnec_coeff,
                           inplace = True)
            
            # Create a copy of the coefficent value in new column
            # as a reference to unmodifed values
            bdb_coeff['REF'] = bdb_coeff[i]
            
            # Modify erronuous values in the interaction columns
            # Replace '<i' , '>i' & .NP,' values with np.nan
            bdb_coeff[i] = bdb_coeff[i].replace(['^>|^<|NV,'] , np.nan,
                                               regex = True)
            
            bdb_coeff.to_csv(write2csv,
                          sep = '\t',
                          index = False)
        
    print('Done')