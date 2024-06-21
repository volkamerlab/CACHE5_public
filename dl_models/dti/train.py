import sys
import hashlib
import pickle
import time
from pathlib import Path
import math

from tqdm import tqdm

import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, RichModelSummary, RichProgressBar
from pytorch_lightning.loggers import WandbLogger

from dl_models.dti.datamodules import PretrainDataModule, MCHR1DataModule
from dl_models.dti.datasets import EnamineDataset
from dl_models.dti.model import CACHE_DTI
from dl_models.dti.utils.mchr import prep_mchr1


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def load_model(batch_size, embed_dim, learning_rate, run_name):
    model = CACHE_DTI(batch_size, embed_dim, learning_rate=learning_rate)
    if run_name is not None:
        run = wandb.init(project="CACHE5")
        # Needs to be adjusted to the correct artifact path, i.e., replace the rindti at the beginning
        artifact = run.use_artifact(f"rindti/CACHE5/model-{run_name}:best", type="model")
        artifact_dir = artifact.download()
        model.load_state_dict(torch.load(Path(artifact_dir) / "model.ckpt")["state_dict"])
        wandb.finish()
    return model


def time2hash(npos):
    # Get the current timestamp
    current_timestamp = str(time.time())

    # Create a SHA-256 hash of the current timestamp
    hash_object = hashlib.sha256(current_timestamp.encode())
    hash_hex = hash_object.hexdigest()

    # Extract the first 4 digits of the hash
    return hash_hex[:npos]


def pre_train(
        base_path: Path = "Pretrain_Data",
        batch_size: int = 128,
        embed_dim: int = 256,
        learning_rate=0.0001
) -> str:
    bdb_path = base_path / "bdb_muts.pkl"
    glass_path = base_path / "glass.pkl"
    mchr1_path = Path("dti_model") / "data" / "mchr1.pkl"
    prep_mchr1(mchr1_path, fold_id=0)

    bdb = PretrainDataModule(bdb_path, batch_size)
    glass = PretrainDataModule(glass_path, batch_size)
    mchr1 = MCHR1DataModule(mchr1_path, batch_size)
    run_name = None
    suffix = "_" + time2hash(4)

    for data_module, name in [(bdb, "BindingDB"), (glass, "GLASS"), (mchr1, "MCHR1")]:
        factor = 1 if name != "MCHR1" else 4
        
        model = load_model(batch_size, embed_dim, learning_rate, run_name)

        logger = WandbLogger(
            log_model="all",
            project="CACHE5",
            name=name.lower() + suffix,
        )
        run_name = logger.experiment.path.split("/")[-1]

        trainer = Trainer(
            callbacks=[
                ModelCheckpoint(save_last=True, mode="min", monitor="val/reg/loss", save_top_k=1),
                RichModelSummary(),
                RichProgressBar(),
            ],
            logger=logger,
            log_every_n_steps=25,
            enable_model_summary=False,
            max_epochs=30 * factor,
        )

        trainer.fit(model, data_module)
        print("(Pre-)Training on", name, "finished")
        if name == "MCHR1":
            trainer.test(ckpt_path="best", datamodule=mchr1)
    return run_name


def predict(run_name, batch_size, embed_dim, learning_rate, out_name=None):
    if out_name is None:
        out_name = "dti_preds.csv"
    mchr1 = MCHR1DataModule(Path("dti_model") / "data" / "mchr1.pkl", batch_size)
    model = load_model(batch_size, embed_dim, learning_rate, run_name)
    
    trainer = Trainer()
    test_pred = trainer.predict(model, mchr1.test_dataloader())
    ensemble_pred = trainer.predict(model, mchr1.predict_dataloader())
    
    with open(out_name, "w") as f:
        print("ID", "cls_prob", "pKi", "pIC50", sep=",", file=f)
        for predictions, subset in [(test_pred, mchr1.test), (ensemble_pred, mchr1.ensemble)]:
            for preds, sample in tqdm(zip([p for preds in predictions for p in preds["full_pred"]], subset)):
                print(
                    sample["ID"], 
                    sigmoid(preds[-1]), 
                    preds[0].item(), 
                    preds[1].item(), 
                    sep=",", 
                    file=f
                )


def develop(run_name=None, out_name=None):
    batch_size, embed_dim = 128, 256
    if run_name is None:
        run_name, mchr1 = pre_train(batch_size, embed_dim)
    else:
        mchr1_path = Path("dti_model") / "data" / "mchr1.pkl"
        mchr1 = MCHR1DataModule(mchr1_path, batch_size)
    predict(run_name, mchr1, batch_size, embed_dim, 0.0001, out_name)


def deploy(data_dir: Path, out_dir: Path, run_name: str, suffix: str = ""):
    batch_size, embed_dim = 128, 256
    ed = EnamineDataset(data_dir / "cleaned_enamine.csv", batch_size)

    file_stump = "dti_screening" + suffix
    if not (screen_file := (out_dir / (file_stump + ".pkl"))).exists():
        model = load_model(batch_size, embed_dim, 0.0001, run_name)
        predictions = Trainer().predict(model, ed.predict_dataloader())
        with open(screen_file, "wb") as f:
            pickle.dump(predictions, f)
    else:
        with open(screen_file, "rb") as f:
            predictions = pickle.load(f)

    with open(out_dir / (file_stump + ".csv"), "w") as f:
        print("ID", "cls_prob", "pKi", "pIC50", sep=",", file=f)
        for pred, sample in tqdm(zip([p for preds in predictions for p in preds["full_pred"]], ed)):
            print(
                sample["ID"], 
                sigmoid(pred[-1]), 
                pred[0].item(),
                pred[1].item(),
                sep=",", 
                file=f
            )


if __name__ == '__main__':
    if sys.argv[1] == "develop":
        if len(sys.argv) >= 4:
            develop(sys.argv[2], sys.argv[3])
        else:
            develop()
    elif sys.argv[1] == "deploy":
        if len(sys.argv) >= 4:
            deploy(Path("Pretrain_Data"), Path("./"), sys.argv[2], sys.argv[3])
        else:
            deploy(Path("Pretrain_Data"), Path("./"), sys.argv[2])
    else:
        print("Unknown command {sys.argv[1]}. Please use \"develop\" or \"deploy\".")

