import pickle
from pathlib import Path
import random

import pandas as pd


def create_dummy_data():
    mols = {
        "L01": "C1CCCCC1",
        "L02": "O1CCOCC1",
        "L03": "C1CCCC2C1CCCC2",
        "L04": "C1CCCCC1C2CCCCC2",
        "L05": "c1ccccc1-c2ccccc2",
        "L06": "CC(C)(C)OC(=O)N1CCC[C@H]1C(=O)O",
        "L07": "O=Cc1ccc(O)c(OC)c1",
        "L08": "CC(=O)NCCC1=CNc2c1cc(OC)cc2",
        "L09": "CCc(c1)ccc2[n+]1ccc3c2[nH]c4c3cccc4",
        "L10": "CCC[C@@H](O)CC\C=C\C=C\C#CC#C\C=C\CO",
    }
    aa_seqs = {f"G{i + 1:02d}": ''.join(random.choices("ACDEFGHIKLMNPQRSTVWY", k=40)) for i in range(10)}

    p_values_range = (5, 10)
    records = {
        "LIG_ID": list(mols.keys()),
        "GENE_ID": list(aa_seqs.keys()),
        "acname": [],
        "acvalue_uM": [],
        "class": [],
    }
    for _ in range(10):
        records["acvalue_uM"].append(round(random.uniform(*p_values_range), 2))
        records["class"].append(int(records["acvalue_uM"][-1] < 6))
        records["acname"].append("pIC50" if random.choice([True, False]) else "pKi")

    data = pd.DataFrame(records)
    with open(Path("dti_model") / "data" / "dummy.pkl", "wb") as f:
        pickle.dump((data, mols, aa_seqs), f)

