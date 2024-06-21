import pickle
from pathlib import Path

import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

from dl_models.dti.utils.t5 import ProtT5


def prepare(root, filename):
    with open(filename, "rb") as f:
        inter, lig_map, tar_map = pickle.load(f)
    if (t5_path := Path(root) / f"prott5_{filename.stem}.pkl").exists():
        with open(t5_path, "rb") as f:
            t5 = pickle.load(f)
    else:
        t5 = ProtT5(list(tar_map.values()))
        with open(t5_path, "wb") as f:
            pickle.dump(t5, f)
    lig_data = {}
    for k, v in lig_map.items():
        try:
            lig_data[k] = torch_geometric.utils.from_smiles(v)
        except:
            pass
    return inter, lig_data, tar_map, t5


def row2data(row, lig_data, tar_map, t5):
    d = lig_data[row["LIG_ID"]].clone()
    d["ID"] = row["LIG_ID"]
    d["t5"] = torch.tensor(t5[tar_map[row["GENE_ID"]]]).reshape(1, -1)
    if row["acname"] == "pKi":
        d["y"] = [row["acvalue_uM"], -1, row["class"]]
    elif row["acname"] == "pIC50":
        d["y"] = [-1, row["acvalue_uM"], row["class"]]
    d["y"] = torch.tensor(d["y"]).reshape(1, -1)
    return d


def load_mchr1_embedding():
    with open(Path("Protein Structures") / "Q99705.fasta") as f:
            mchr1_seq = "".join(l.strip() for l in f.readlines()[1:])
    
    if (t5_path := Path("dti_model") / "data" / "prott5_mchr1.pkl").exists():
        with open(t5_path, "rb") as f:
            return pickle.load(f)[mchr1_seq]
    else:
        t5 = ProtT5([mchr1_seq])
        with open(t5_path, "wb") as f:
            pickle.dump(t5, f)
        return t5[mchr1_seq]


class BaseDataset(InMemoryDataset):
    def __init__(self, filename, path_idx: int = 0):
        self.filename = Path(filename)
        super().__init__(Path("dti_model") / "data")
        self.data, self.slices = torch.load(self.processed_paths[path_idx])

    @property
    def processed_paths(self):
        return [self.root / f for f in self.processed_file_names]

    def process_(self, data, path_idx: int = 0):
        data, slices = self.collate(data)
        torch.save((data, slices), self.processed_paths[path_idx])


class PretrainDataset(BaseDataset):
    def __init__(self, filename):
        super().__init__(filename)

    @property
    def processed_file_names(self):
        return [self.filename.stem + ".pt"]

    def process(self):
        inter, lig_data, tar_map, t5 = prepare(self.root, self.filename)
        data = []
        for _, row in inter.iterrows():
            try:
                data.append(row2data(row, lig_data, tar_map, t5))
            except Exception as e:
                # pass
                print(e)
        self.process_(data)


class EnamineDataset(PretrainDataset):
    def __init__(self, filename, batch_size):
        self.filename = Path(filename)
        super().__init__(Path(filename))
        self.batch_size = batch_size
    
    def predict_dataloader(self):
        return DataLoader(self, batch_size=self.batch_size, shuffle=False)

    def process(self):
        t5_embed = load_mchr1_embedding()
        data = []
        with open(self.filename, "r") as f:
            for line in f:
                try:
                    idx, smiles = line.strip().split(",")
                    d = torch_geometric.utils.from_smiles(smiles)
                    d["ID"] = idx
                    d["t5"] = t5_embed.clone()
                    data.append(d)
                except Exception as e:
                    raise e
        self.process_(data)


class MCHR1Dataset(BaseDataset):
    splits = {"train": 0, "val": 1, "ensemble": 2, "test": 3}

    def __init__(self, filename, split):
        super().__init__(filename, self.splits[split])

    @property
    def processed_file_names(self):
        return [k + ".pt" for k in ["train", "val", "ensemble", "test"]]

    def process(self):
        inter, lig_data, tar_map, t5 = prepare(self.root, self.filename)
        data = {k: [] for k in self.splits.keys()}
        for _, row in inter.iterrows():
            try:
                data[row["split"]].append(row2data(row, lig_data, tar_map, t5))
            except Exception as e:
                print(e)
        for split in self.splits.keys():
            self.process_(data[split], self.splits[split])
