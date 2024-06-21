import pickle

import torch
from pytorch_lightning import LightningModule
from torch import nn, Tensor
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torchmetrics import MetricCollection, Accuracy, AUROC, AveragePrecision, MatthewsCorrCoef, MeanAbsoluteError, \
    MeanSquaredError, ExplainedVariance, SpearmanCorrCoef, PearsonCorrCoef


class MolEmbed(LightningModule):
    def __init__(self, input_dim: int, output_dim: int = 128, num_layers: int = 3):
        """hidden_dim=output_dim"""
        super().__init__()
        print(input_dim, output_dim)
        self.inp = GINConv(
            nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.PReLU(),
                nn.Linear(output_dim, output_dim),
                nn.BatchNorm1d(output_dim),
            )
        )
        mid_layers = [
            GINConv(
                nn.Sequential(
                    nn.Linear(output_dim, output_dim),
                    nn.PReLU(),
                    nn.Linear(output_dim, output_dim),
                    nn.BatchNorm1d(output_dim),
                )
            )
            for _ in range(num_layers - 2)
        ]
        self.mid_layers = nn.ModuleList(mid_layers)
        self.out = GINConv(
            nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.PReLU(),
                nn.Linear(output_dim, output_dim),
                nn.BatchNorm1d(output_dim),
            )
        )

    def forward(self, data, **kwargs) -> Tensor:
        """Forward the data through the GNN module"""
        # print(data)
        x = self.inp(data.x, data.edge_index)
        for module in self.mid_layers:
            x = module(x, data.edge_index)
        x = self.out(x, data.edge_index)
        pool = global_mean_pool(x, data["batch"])
        return F.normalize(pool, dim=1)


class CACHE_DTI(LightningModule):
    def __init__(self, batch_size, output_dim, learning_rate=0.0001, **kwargs):
        super().__init__()
        self.drug_embedder = MolEmbed(input_dim=9, output_dim=output_dim, num_layers=3)
        self.prot_embedder = nn.Sequential(nn.Linear(768, output_dim), nn.PReLU())
        self.head = nn.Sequential(
            nn.Linear(2 * output_dim, output_dim // 2),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_dim // 2, 64),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3),
        )
        self.lr = learning_rate
        self.reg_criterion = nn.MSELoss()
        self.class_criterion = nn.BCEWithLogitsLoss()
        self.metrics = self._set_metrics()
        self.automatic_optimization = False
        self.batch_size = batch_size

    def to(self, device):
        super().to(device)
        for splits in self.metrics.values():
            for mc in splits.values():
                mc.to(device)

    def stable_bcewl(self, pred, target, epsilon=1e-10):
        stable_bce_loss = -target * torch.log(pred + epsilon) - (1 - target) * torch.log(1 - pred + epsilon)
        return stable_bce_loss.mean()

    def _set_metrics(self):
        class_metrics = MetricCollection([
            Accuracy(task="binary"),
            AUROC(task="binary"),
            AveragePrecision(task="binary"),
            MatthewsCorrCoef(task="binary"),
        ])
        reg_metrics = MetricCollection([
            MeanAbsoluteError(),
            MeanSquaredError(),
            ExplainedVariance(),
            PearsonCorrCoef(),
            SpearmanCorrCoef(),
        ])
        return {
            "class": {
                "train": class_metrics.clone("train/class/"),
                "val": class_metrics.clone("val/class/"),
                "test": class_metrics.clone("test/class/"),
            },
            "reg": {
                "train": reg_metrics.clone("train/reg/"),
                "val": reg_metrics.clone("val/reg/"),
                "test": reg_metrics.clone("test/reg/"),
            }
        }
    
    def check_nans(self, tensor, name=""):
        if torch.isnan(tensor).any():
            print(f"NaN detected in {name}")
            return True
        return False

    def forward(self, data):
        try:  # Necessary as inconsistencies in returned batch-sizes by the DrugEmbedder were encountered (only 127 instead of 128 embeddings)
            # print("=" * 15, "START", "=" * 15)
            # self.check_nans(data["x"].float(), "x")
            # self.tensor_stats(data["x"].float(), "x")
            # self.check_nans(data["t5"], "T5")
            # self.tensor_stats(data["t5"], "T5")
            drug_embed = self.drug_embedder(data)
            prot_embed = self.prot_embedder(data["t5"])
            # self.check_nans(drug_embed, "Drug")
            # self.tensor_stats(drug_embed, "Drug")
            # self.check_nans(prot_embed, "Prot")
            # self.tensor_stats(prot_embed, "Prot")
            comb_embed = torch.cat((drug_embed, prot_embed), dim=1)
            # self.check_nans(comb_embed, "Comb")
            # self.tensor_stats(comb_embed, "Comb")
            pred = self.head(comb_embed)
            # self.check_nans(pred, "Pred")
            # self.tensor_stats(pred, "Pred")
            mask = data["y"][:, :2] != -1
            full_pred = pred.detach().cpu().numpy().copy()
            data["y"] = torch.nan_to_num(data["y"], 4, 4, 4)
            return {
                "reg_pred": pred[:, :-1],
                "reg_labels": data["y"][:, :-1],
                "class_pred": pred[:, -1],
                "class_labels": data["y"][:, -1].long(),  # .float(),
                "full_pred": full_pred,
                "mask": mask,
            }
        except Exception as e:
            # raise e
            print("Exception:", e)
            return None

    def shared_step(self, data):
        fwd = self.forward(data)
        if fwd is None:
            return None
        
        # print(fwd_dict["reg_pred"][fwd_dict["mask"]][:5], fwd_dict["reg_labels"][fwd_dict["mask"]][:5])
        # print(fwd["reg_pred"][fwd["mask"]])
        # print(fwd["reg_labels"][fwd["mask"]])
        # print((fwd["reg_pred"][fwd["mask"]] - fwd["reg_labels"][fwd["mask"]]) ** 2)
        fwd["reg_loss"] = self.reg_criterion(fwd["reg_pred"][fwd["mask"]], fwd["reg_labels"][fwd["mask"]]) / 10
        fwd["class_loss"] = self.class_criterion(fwd["class_pred"], fwd["class_labels"].float())
        # fwd["class_loss"] = self.stable_bcewl(fwd["class_pred"], fwd["class_labels"].float())
        # self.check_nans(fwd["reg_loss"], "Reg Loss")
        # self.tensor_stats(fwd["reg_loss"], "Reg Loss")
        # print(fwd["reg_loss"])
        return fwd

    def update(self, fwd, stage):
        self.metrics["reg"][stage].update(fwd["reg_pred"].contiguous()[fwd["mask"]], fwd["reg_labels"].contiguous()[fwd["mask"]])
        self.metrics["class"][stage].update(fwd["class_pred"], fwd["class_labels"])
        # print(fwd["reg_loss"], fwd["class_loss"], sep="\n")
        self.log(f"{stage}/reg/loss", fwd["reg_loss"], batch_size=self.batch_size)
        self.log(f"{stage}/class/loss", fwd["class_loss"], batch_size=self.batch_size)

    def tensor_stats(self, tensor, name):
        if tensor is not None:
            print(f"{name}: mean={tensor.mean().item()}, std={tensor.std().item()}, max={tensor.max().item()}, min={tensor.min().item()}")

    def training_step(self, data, data_idx):
        fwd = self.shared_step(data)
        if fwd is None:
            return None
        self.update(fwd, "train")

        # Do backpropagation
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(fwd["reg_loss"], retain_graph=True)
        self.manual_backward(fwd["class_loss"])
        # nan_found = False
        # for name, param in self.named_parameters():
        #     if param.grad is not None:
        #         nan_found = nan_found or self.check_nans(param.grad, f"grad {name}")
        #         self.tensor_stats(param.grad, f"grad {name}")
        self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        opt.step()
        # for name, param in self.named_parameters():
        #     nan_found = nan_found or self.check_nans(param, f"param {name}")
        #     self.tensor_stats(param, f"param {name}")
        # print("=" * 16, "END", "=" * 16)
        # if nan_found:
        #     exit(0)

        return fwd

    def validation_step(self, data, data_idx):
        fwd = self.shared_step(data)
        if fwd is None:
            return None
        self.update(fwd, "val")
        return fwd

    def test_step(self, data, data_idx):
        fwd = self.shared_step(data)
        if fwd is None:
            return None
        self.update(fwd, "test")
        return fwd

    def predict_step(self, data, data_idx):
        with open("dti_model/data/prott5_mchr1.pkl", "rb") as f:
            data["t5"] = torch.tensor(list(pickle.load(f).values())[0]).to("cuda:0").reshape(1, -1)
        drug_embed = self.drug_embedder(data)
        prot_embed = self.prot_embedder(data["t5"]).repeat(drug_embed.shape[0], 1)
        comb_embed = torch.cat((drug_embed, prot_embed), dim=1)
        pred = self.head(comb_embed)
        return {"full_pred": pred}
        
    def log_all(self, metrics: dict):
        for k, v in metrics.items():
            self.log(k, v)

    def shared_end(self, stage):
        for task in ["reg", "class"]:
            metrics = self.metrics[task][stage].compute()
            self.metrics[task][stage].reset()
            self.log_all(metrics)

    def on_train_epoch_end(self):
        self.shared_end("train")

    def on_validation_epoch_end(self):
        self.shared_end("val")

    def on_test_epoch_end(self):
        self.shared_end("test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)  # , lr=self.lr)

