import argparse
import csv
import os
import pickle
import pathlib
import time

import pandas as pd
from sklearn.utils import gen_batches

from tqdm import tqdm
from molbert.utils.featurizer.molbert_featurizer import MolBertFeaturizer

ROOT_DIR = pathlib.Path(__file__).parents[2]


def screen_molbert(
        data_dir,
        classifier_path,
        output_dir: str,
        batch_size=4096,
):
    # Model loading
    molbert_ckpt = ROOT_DIR / "checkpoints" /"molbert_100epochs" / "checkpoints" / "last.ckpt"
    if not molbert_ckpt.exists() or not molbert_ckpt.is_file():
        raise ValueError(f"MolBERT checkpoint at {molbert_ckpt} not found. Please check the README on how to download"
                         "the required files.")
    molbert = MolBertFeaturizer(str(molbert_ckpt))
    with open(classifier_path, "rb") as f:
        clf = pickle.load(f)

    # Load data
    if isinstance(data_dir, str):
        data_file = os.path.join(data_dir, "cleaned_enamine.csv")
    elif isinstance(data_dir, pathlib.Path):
        data_file = str(data_dir / "cleaned_enamine.csv")
    else:
        raise ValueError(f"Expected pathlib.Path or str for 'data_dir', got {type(data_dir)}.")

    df = pd.read_csv(data_file, names=["TID", "SMILES"])

    # Remove header row if present.
    if df.iloc[0].tolist == ["TID", "SMILES"]:
        df = df.iloc[1:]

    # Screen in batches and directly save to file to prevent high memory usage
    filename = os.path.join(str(output_dir), f"screening_molbert.csv")
    print(f"{filename}: Processing ({len(df)}) molecules.", flush=True)
    with open(filename, "w") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(["ID", "pIC50", "pKi", "cls_prb"])
        for batch_idx in tqdm(gen_batches(len(df), batch_size), desc=filename):
            samples = df.iloc[batch_idx]
            embeddings, _ = molbert.transform(samples["SMILES"])
            cls_prb = clf.predict_proba(embeddings)[:, 1]

            writer.writerows([
                (id_, None, None, p) for id_, p in zip(samples["TID"].tolist(), cls_prb.squeeze())
            ])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("enamine_split", help="The chunk of enamine to process by this job.", type=pathlib.Path)
    args = parser.parse_args()
    screen_molbert(
        classifier_path= ROOT_DIR / "dl_models" / "molbert" / "molbert_classifier.pkl",
        output_dir=ROOT_DIR / "screening" / args.enamine_split.stem
    )