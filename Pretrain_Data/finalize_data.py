from pathlib import Path
import pickle

import pandas as pd
import numpy as np


def collect_glass():
    base = Path("GLASS") / "processed"

    ic = pd.read_csv(base / "interaction_tables" / "GLASS_IC50.tsv", sep="\t").rename(columns={"VALUE": "acvalue_uM"})
    ic["acname"] = "pIC50"
    
    ki = pd.read_csv(base / "interaction_tables" / "GLASS_KI.tsv", sep="\t").rename(columns={"VALUE": "acvalue_uM"})
    ki["acname"] = "pKi"
    
    ligand = pd.read_csv(base / "GLASS_Ligand_Fixed.tsv", sep="\t")
    ligand_map = dict(ligand[["LIG_ID", "GLASS_ISO_SMILES"]].values)
    
    target = pd.read_csv(base / "GLASS_Target_Fixed.tsv", sep="\t")
    target_map = dict(target[["GENE_ID", "FASTA"]].values)
    
    cmb = pd.concat([ic, ki])
    cmb["acvalue_uM"] = cmb["acvalue_uM"].apply(lambda x: 9 - np.log(x))
    cmb["class"] = cmb["acvalue_uM"].apply(lambda x: int(x < 6))
    
    print(f"Saving {len(cmb)} interactions from BindingDB.")
    with open("glass.pkl", "wb") as f:
        pickle.dump((cmb, ligand_map, target_map), f)


def collect_bdb():
    base = Path("BindingDB") / "processed" / "no_muts"

    ic = pd.read_csv(base / "mN_interactions" / "BDB_mN_IC50.tsv", sep="\t")
    ic = ic[["LIG_ID", "GENE_ID", "IC50"]].dropna(axis="index").rename(columns={"IC50": "acvalue_uM"})
    ic["acname"] = "pIC50"
    
    ki = pd.read_csv(base / "mN_interactions" / "BDB_mN_KI.tsv", sep="\t")
    ki = ki[["LIG_ID", "GENE_ID", "KI"]].dropna(axis="index").rename(columns={"KI": "acvalue_uM"})
    ki["acname"] = "pKI"
    
    ligand = pd.read_csv(base / "BDB_mN_Ligand_Clean.tsv", sep="\t")
    ligand_map = dict(ligand[["LIG_ID", "BDB_SMILES"]].values)
    
    target = pd.read_csv(base / "BDB_mN_Target_Clean.tsv", sep="\t")
    target_map = dict(target[["GENE_ID", "BDB_CHAIN_SEQ"]].values)
    
    cmb = pd.concat([ic, ki])
    cmb["acvalue_uM"] = cmb["acvalue_uM"].apply(lambda x: 9 - np.log(x))
    cmb["class"] = cmb["acvalue_uM"].apply(lambda x: int(x < 6))
    
    print(f"Saving {len(cmb)} interactions from BindingDB.")
    with open("bdb_muts.pkl", "wb") as f:
        pickle.dump((cmb, ligand_map, target_map), f)


if __name__ == "__main__":
    collect_glass()
    collect_bdb()
    
