import json
import os
import tempfile

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils import gen_batches

from molbert.apps.finetune import FinetuneSmilesMolbertApp
import evaluate
from molbert.utils.featurizer.molbert_featurizer import MolBertFeaturizer

# The below ENV declaration was needed for Mac
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["WANDB_PROJECT"] = "cache5_molformer"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"


def compute_metrics(eval_pred, metric="roc_auc"):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metric_func = evaluate.load(metric)
    return metric_func.compute(prediction_scores=predictions, references=labels)


def frozen_tuning(
    model: MolBertFeaturizer,
    train_smiles,
    train_targets,
    test_smiles,
    test_targets,
    ft,
    performance,
):
    train_rep = []
    for batch in gen_batches(len(train_smiles), 512):
        descriptors, valid = model.transform(train_smiles[batch])
        assert valid.all()
        train_rep.append(descriptors)

    test_rep = []
    for batch in gen_batches(len(test_smiles), 512):
        descriptors, valid = model.transform(test_smiles[batch])
        assert valid.all()
        test_rep.append(descriptors)

    train_rep = np.concatenate(train_rep)
    test_rep = np.concatenate(test_rep)
    for ft_method, clf in ft.items():
        clf.fit(train_rep, train_targets)
        preds = clf.predict(test_rep)
        performance[ft_method].append(metrics.roc_auc_score(test_targets, preds))


def cross_validation(
    model_ckpt,
    model_frozen,
    df,
    num_folds=8,
    fold_col="DataSAIL_10f",
    smiles_col="smiles",
    class_col="class",
    tune=True,
    ml_dict=None,
    num_epoch=10,
):
    performance = {}
    if ml_dict is not None:
        for ml in ml_dict.keys():
            performance[ml] = []
    if tune:
        performance[f"fine_tuned_{num_epoch}"] = []

    for fold_idx in range(num_folds):
        test_fold = fold_idx
        val_fold = fold_idx + 1 if fold_idx != num_folds - 1 else 0
        test_idx = df[df[fold_col] == f"Fold_{test_fold}"].index
        test_smiles = df.loc[test_idx][smiles_col]
        test_targets = df.loc[test_idx][class_col]

        val_idx = df[df[fold_col] == f"Fold_{val_fold}"].index
        val_smiles = df.loc[val_idx][smiles_col]
        val_targets = df.loc[val_idx][class_col]

        train_idx = df[
            ~df[fold_col].isin([f"Fold_{test_fold}", f"Fold_{val_fold}"])
        ].index
        train_smiles = df.loc[train_idx][smiles_col]
        train_targets = df.loc[train_idx][class_col]

        if ml_dict is not None:
            print("Cross validation with frozen model")
            frozen_tuning(
                model_frozen,
                train_smiles.to_list() + val_smiles.to_list(),
                train_targets.to_list() + val_targets.to_list(),
                test_smiles.to_list(),
                test_targets.to_list(),
                ml_dict,
                performance,
            )
        if tune:
            # FinetuneSmilesMolbertApp wants the data as files
            with tempfile.TemporaryDirectory() as tdir:
                train_file = os.path.join(tdir, "train.csv")
                val_file = os.path.join(tdir, "val.csv")
                test_file = os.path.join(tdir, "test.csv")

                # MolBERT hardcodes the smiles column as SMILES
                df.loc[train_idx][[smiles_col, class_col]].rename(
                    columns={"smiles": "SMILES"}
                ).to_csv(train_file)
                df.loc[val_idx][[smiles_col, class_col]].rename(
                    columns={"smiles": "SMILES"}
                ).to_csv(val_file)
                df.loc[test_idx][[smiles_col, class_col]].rename(
                    columns={"smiles": "SMILES"}
                ).to_csv(test_file)

                raw_args_str = (
                    f"--batch_size 32 "
                    f"--max_epochs 10 "
                    f"--train_file {train_file} "
                    f"--valid_file {val_file} "
                    f"--test_file {test_file} "
                    f"--mode classification "
                    f"--output_size 2 "
                    f"--pretrained_model_path {model_ckpt} "
                    f"--smiles_column SMILES "  # Has no effect for the internal MolBERT dataset
                    f"--label_column {class_col} "
                    f"--freeze_level 1 "
                    f"--learning_rate 0.0001 "
                    f"--learning_rate_scheduler linear_with_warmup "
                    f"--wandb"
                )
                raw_args = raw_args_str.split(" ")

                trainer = FinetuneSmilesMolbertApp().run(raw_args)
                trainer.test(ckpt_path="best")

                metrics_path = os.path.join(
                    os.path.dirname(trainer.ckpt_path), "metrics.json"
                )
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)

                performance[f"fine_tuned_{num_epoch}"].append(metrics["AUROC"])
            # performance[f'fine_tuned_{num_epoch}'].append(auroc)
        print(f"fold_{fold_idx}", performance)
    performance_df = pd.DataFrame(performance)
    performance_df.to_csv("molformer_auroc.csv")


if __name__ == "__main__":
    d_path = "../../Reference Data/20240430_MCHR1_splitted_RJ.csv"
    df = pd.read_csv(d_path, index_col=0)
    # These three molecules are significantly bigger than the rest of the molecules.
    # They also exceed the maximum length of molbert leading to NaNs
    df.drop([1645, 1646, 1647], axis=0, inplace=True)
    df = df.reset_index()

    ckpt = "../../checkpoints/molbert_100epochs/checkpoints/last.ckpt"
    model_frozen = MolBertFeaturizer(ckpt)
    frozen_ml = {"svm": SVC(), "rf": RandomForestClassifier()}

    num_labels = len(df["class"].unique())

    cross_validation(
        ckpt, model_frozen, df, ml_dict=frozen_ml, num_epoch=10, tune=True, num_folds=10
    )
