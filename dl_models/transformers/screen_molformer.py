import csv
import pathlib
import time
from datasets import Dataset
import pandas as pd
import torch
from torch.nn.functional import softmax
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import os
from sklearn.utils import gen_batches

from dl_models.transformers.multi_objective import MolformerForMultiobjective


os.environ["TOKENIZERS_PARALLELISM"] = "true"
ROOT_DIR = pathlib.Path(__file__).parents[2]
BATCH_SIZE = 4096


def screen_molformer(
    data_dir,
    ckpt_path,
    out_dir: str
):
    reference_df = pd.read_csv(
        ROOT_DIR / "Reference Data" / "20240430_MCHR1_splitted_RJ.csv"
    )
    test_df = Dataset.from_pandas(
        reference_df[
            (reference_df["DataSAIL_10f"] == "Fold_8")
            | (reference_df["DataSAIL_10f"] == "Fold_9")
        ][["smiles", "ID"]]
    ).rename_column("smiles", "text")

    if isinstance(data_dir, str):
        enamine_file = os.path.join(data_dir, "cleaned_enamine.csv")
    elif isinstance(data_dir, pathlib.Path):
        enamine_file = str(data_dir / "cleaned_enamine.csv")

    df = pd.read_csv(enamine_file)

    tokenizer = AutoTokenizer.from_pretrained(
        "ibm/MoLFormer-XL-both-10pct", trust_remote_code=True, use_fast=True
    )
    tokenizer_config = {
        "padding": "max_length",
        "truncation": True,
        "max_length": tokenizer.model_max_length,
        "return_tensors": "pt",
    }

    def tokenize_input(examples):
        return tokenizer(examples["text"], **tokenizer_config)

    # Data for ensembling
    tokenized_test = test_df.map(tokenize_input, batched=True)
    val_loader = DataLoader(tokenized_test, shuffle=False, batch_size=1_000)

    model = MolformerForMultiobjective.from_pretrained(
        ckpt_path,
    )
    model.eval()
    model = model.to("cuda")

    with torch.no_grad():
        # Predictions on the test set
        with open(os.path.join(str(out_dir), "molformer_ensemble_predictions.csv"), "w") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow(["ID", "pIC50", "pKi", "cls_prb"])
            for batch in tqdm(val_loader):
                ids = batch.pop("ID")
                batch = {
                    k: torch.stack(v).T
                    for k, v in batch.items()
                    if k in ["input_ids", "attention_mask"]
                }
                batch = {k: v.to("cuda") for k, v in batch.items()}
                outputs = model(**batch)
                y_hat, logits = outputs.logits
                y_hat = y_hat.cpu().numpy().squeeze()
                probs = softmax(logits, dim=1).cpu().numpy()[:, 1]

                writer.writerows(
                    [(id_, y, y, p) for id_, y, p in zip(ids, y_hat, probs)]
                )

        # Predictions on enamine
        with open(ROOT_DIR / "screening.csv", "w") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow(["ID", "pIC50", "pKi", "cls_prb"])

            for batch_idx in tqdm(gen_batches(len(df), BATCH_SIZE)):
                samples = df.iloc[batch_idx]
                t0 = time.time()
                # Tokenizing the whole dataset consumes too much memory
                tokens = tokenizer(samples["SMILES"].tolist(), **tokenizer_config)
                print(f"Tokenized ({BATCH_SIZE}) molecules in {time.time() - t0:.3f}s.")
                tokens = {k: v.to("cuda") for k, v in tokens.items()}

                outputs = model(**tokens)
                y_hat, logits = outputs.logits
                y_hat = y_hat.cpu().numpy().squeeze()
                probs = softmax(logits, dim=1).cpu().numpy()[:, 1]

                writer.writerows(
                    [(id_, y, y, p) for id_, y, p in zip(samples["TID"], y_hat, probs)]
                )


if __name__ == "__main__":
    screen_molformer(
        ROOT_DIR / "cleaned_enamine.csv",
        "/data/users/mrdupont/Cache5MolFormer-Finetuning/checkpoints/molformer_cv_models/multiobjective_0.8_tuned/cv_6/",
        out_dir=str(ROOT_DIR)
    )
