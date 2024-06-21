import pickle
from pathlib import Path

import pandas as pd


def fold2split(fold_name, fold_id):
    fold_nr = int(fold_name[-1])
    if fold_nr == 8:
        return "ensemble"
    if fold_nr == 9:
        return "test"
    if fold_nr == (fold_id + 6) % 7:
        return "val"
    return "train"


def prep_mchr1(output_path, fold_id):
    df = pd.read_csv(Path("Reference Data") / "20240430_MCHR1_splitted_RJ.csv")

    with open(Path("Protein Structures") / "Q99705.fasta") as f:
        tar_map = {"P1": "".join(l.strip() for l in f.readlines()[1:])}
    lig_map = dict(df[["ID", "smiles"]].values)

    df["GENE_ID"] = "P1"
    df.rename(columns={"ID": "LIG_ID"}, inplace=True)
    df["split"] = df["DataSAIL_10f"].apply(lambda x: fold2split(x, fold_id))
    inter = df[["LIG_ID", "GENE_ID", "acvalue_uM", "acname", "class", "split"]]

    with open(output_path, "wb") as f:
        pickle.dump((inter, lig_map, tar_map), f)
