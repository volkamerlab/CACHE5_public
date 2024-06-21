#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 18:40:03 2024

@author: vka24
"""

from GLASS import glass_main1
from fix_glass_ligands import glass_main2
from fix_glass_targets import glass_main3
from fix_glass_interacts import glass_main4
from separate_glass_interactions import glass_main5
from convert_glass_units import glass_main6
from create_glass_tables import glass_main7
from create_glass_matrix import glass_main8

# Clean Dataset
tar_loc = 'raw/GLASS_targets.tsv'
lig_loc = 'raw/GLASS_ligands.tsv'
int_loc = 'raw/GLASS_interactions_active.tsv'

print('\nRunning Glass.py')
glass_main1(tar_loc , lig_loc , int_loc)
print('Run complete\n')


# Fix dataset problems
print('Running fix_glass_ligands.py')
glass_main2()
print('Run complete\n')


print('Running fix_glass_targets.py')
glass_main3()
print('Run complete\n')


print('Running fix_glass_interacts.py')
glass_main4()
print('Run complete\n')


# Separating interactions based on affinity
# Files related to parameters in coeff will be in their respective folders
# in processed/interactions/
# All else will be in  others
print('Running separate_glass_interactions.py')
glass_main5()
print('Run complete\n')


# Converting units to uM in KI, KD, IC50 & EC50 which are possible
# values with ynits such as "%" dropped.
# Combines the multiple datasets for each parameter
print('Running convert_glass_units.py')
glass_main6()
print('Run complete\n')


# Grouping the interactions based on median, removing values > 10000nM
print('Running convert_glass_table.py')
glass_main7()
print('Running complete\n')


# Matrix creation
print('Running create_glass_matrix.py')
glass_main8()
print('Run complete\n')
