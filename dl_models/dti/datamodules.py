import torch
from pytorch_lightning import LightningDataModule
from torch_geometric.loader import DataLoader

from dl_models.dti.datasets import PretrainDataset, MCHR1Dataset


class BaseDataModule(LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=min(self.batch_size, len(self.train)), shuffle=True)

    def val_dataloader(self):
        dl = DataLoader(self.val, batch_size=min(self.batch_size, len(self.val)), shuffle=False)
        return dl

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=min(self.batch_size, len(self.test)), shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.ensemble, batch_size=min(self.batch_size, len(self.ensemble)), shuffle=False)


class PretrainDataModule(BaseDataModule):
    def __init__(self, filename, batch_size):
        super().__init__(batch_size)
        ds = PretrainDataset(filename)
        self.train, self.val = torch.utils.data.dataset.random_split(ds, [0.8, 0.2])


class MCHR1DataModule(BaseDataModule):
    def __init__(self, filename, batch_size):
        super().__init__(batch_size)

        self.train = MCHR1Dataset(filename, "train")
        self.val = MCHR1Dataset(filename, "val")
        self.ensemble = MCHR1Dataset(filename, "ensemble")
        self.test = MCHR1Dataset(filename, "test")
