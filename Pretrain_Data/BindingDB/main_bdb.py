# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 18:46:33 2024

@author: HP
"""

from BindingDB import bdb_main1
from fix_bdb_targets import bdb_main2
from fix_bdb_ligands import bdb_main3
from fix_bdb_interacts1 import bdb_main4
from separate_interacts import bdb_main5
from fix_bdb_interacts2 import bdb_main6
from create_bdb_matrix import bdb_main7


# Separating & Cleaning complete dataset
df_loc = "raw/BindingDB_All_202403.tsv"
chunk_size = 1000000

print('Running BindingDB.py')
bdb_main1(df_loc , chunk_size)
print('Run complete\n')


# Fixing the Target dataset
print('Running fix_bdb_targets.py')
bdb_main2()
print('Run complete\n')


# Fixing the Ligand dataset
print('Running fix_bdb_ligands.py')
bdb_main3()
print('Run complete\n')


# First round of interaction fixing
print('Running fix_bdb_interacts1.py')
bdb_main4()
print('Run complete\n')


# Separating the interactions based on interaction coefficients
print('Running separate_interacts.py')
bdb_main5()
print('Run complete\n')


# Cleaning separated interaction data
print('Running fix_bdb_interacts2.py')
bdb_main6()
print('Run complete\n')


#Creating matrix
print('Running create_bdb_matrix.py')
bdb_main7()
print('Run complete\n')