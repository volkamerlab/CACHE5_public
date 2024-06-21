import os
import pathlib
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import gen_batches
from tqdm import tqdm

from molbert.utils.featurizer.molbert_featurizer import MolBertFeaturizer


def find_model_params(targets, fingerprints, physchem_descriptors, molbert_embeddings, folds):
    model_params = {}
    for descriptor_name, inputs in zip(["Morgan FP", "PhysChem", "MolBERT"],
                                       [fingerprints, physchem_descriptors, molbert_embeddings]):
        if not inputs:
            continue

        for model_name in ["rf", "mlp"]:
            if model_name == "rf":
                search = GridSearchCV(
                    RandomForestClassifier(random_state=42),
                    param_grid={
                        'n_estimators': [50, 100, 250, 500],
                        'max_features': ['sqrt', 'log2'],
                        'max_depth': [20, 50, None]
                    }, cv=folds, scoring='roc_auc', verbose=10, n_jobs=-1,
                )
            elif model_name == "mlp":
                search = GridSearchCV(
                    MLPClassifier(solver="adam", random_state=42),
                    param_grid={
                        'hidden_layer_sizes': [(100, 100), (100, 100, 100), (512, 512, 512)],
                        'alpha': [0.0001, 0.001, 0.01]
                    }, cv=folds, scoring='roc_auc', verbose=10, n_jobs=-1,
                )
            search.fit(inputs, targets)
            model_params[(model_name, descriptor_name)] = search.best_params_

    return model_params


def get_model(model_name, descriptor_name, params):
    if model_name == "rf":
        return RandomForestClassifier(random_state=42, **params[(model_name, descriptor_name)])
    elif model_name == "mlp":
        return MLPClassifier(solver="adam", random_state=42, **params[(model_name, descriptor_name)])


def evaluate(targets, fingerprints, physchem_descriptors, molbert_embeddings, folds):
    if not os.path.exists("classification_params.pkl"):
        model_params = find_model_params(targets, fingerprints, physchem_descriptors, molbert_embeddings, folds=folds)
        with open("classification_params.pkl", 'wb') as f:
            pickle.dump(model_params, f)
    else:
        with open("classification_params.pkl", "rb") as f:
            model_params = pickle.load(f)

    results = []
    for descriptor_name, inputs in zip(["Morgan FP", "PhysChem", "MolBERT"],
                                       [fingerprints, physchem_descriptors, molbert_embeddings]):
        for j, model_name in enumerate(["rf", "mlp"]):
            model = get_model(model_name, descriptor_name, model_params)

            if descriptor_name == "PhysChem":  # Features may vary a lot in value
                pipe = make_pipeline(StandardScaler(), model)
            else:
                pipe = model

            output = cross_validate(pipe, inputs, targets, cv=folds, scoring="roc_auc", verbose=1, n_jobs=8,
                                    return_train_score=True, return_estimator=True, return_indices=True)
            output["descriptor"] = descriptor_name
            output["model"] = model_name
            results.append(output)

    df = pd.DataFrame(results)

    return df.explode(["test_score", "fit_time", "score_time", "train_score", "estimator"], ignore_index=True)


def train_molbert(
    predictions_out_path = "molbert_ensemble_predictions.csv",
    model_out_path = "molbert_classifier.pkl"
):
    here = pathlib.Path(__file__).parent.absolute()
    # Load the data
    data = pd.read_csv(here.parent / "Reference Data" / "20240430_MCHR1_splitted_RJ.csv", index_col=0)

    # These three molecules are significantly bigger than the rest of the molecules. They are too long
    # For MolBERT to correctly tokenize
    data.drop([1645, 1646, 1647], axis=0, inplace=True)
    data = data.reset_index()
    molbert_ckpt = here.parent / "checkpoints" / "molbert_100epochs" / "checkpoints" / "last.ckpt"
    if not molbert_ckpt.exists() or not molbert_ckpt.is_file():
        raise ValueError(f"MolBERT checkpoint at {molbert_ckpt} not found. Please check the README on how to download"
                         "the required files.")

    molbert = MolBertFeaturizer(str(molbert_ckpt))

    # Data splits
    validation_fold_mask = data["DataSAIL_10f"] == 'Fold_8'
    test_fold_mask = data["DataSAIL_10f"] == 'Fold_9'
    training_fold_mask = ~data["DataSAIL_10f"].isin(["Fold_8", "Fold_9"])

    validation_fold = data.loc[validation_fold_mask]
    test_fold = data.loc[test_fold_mask]
    training_fold = data.loc[training_fold_mask]

    K = len(training_fold["DataSAIL_10f"].unique())
    cv_folds = []
    for i in range(K):
        cv_test_fold = training_fold[training_fold["DataSAIL_10f"] == f"Fold_{i}"].index
        cv_train_fold = training_fold[training_fold["DataSAIL_10f"] != f"Fold_{i}"].index
        cv_folds.append((cv_train_fold, cv_test_fold))

    # Features
    torch.manual_seed(42)
    np.random.seed(42)
    # fps = np.stack([compute_descriptors(smiles, method="Morgan FPs") for smiles in tqdm(data["smiles"])])
    # descriptors = np.array([compute_descriptors(smiles, method="PhysChem") for smiles in tqdm(data["smiles"])])
    embeddings = np.concatenate([molbert.transform(data['smiles'][batch])[0] for batch in tqdm(gen_batches(len(data['smiles']), 512), total=(len(data)//512)+1)])

    # Train models with MolBERT features and store best model & predictions
    results_df = evaluate(data["class"], fps=[], descriptors=[], molbert_embeddings=embeddings, folds=cv_folds)
    best_classifiers = results_df.loc[results_df.groupby("descriptor")["test_score"].idxmax()]

    molbert_clf = best_classifiers.query("descriptor == 'MolBERT'")["estimator"].iloc[0]
    val_preds = molbert_clf.predict_proba(embeddings[validation_fold_mask])[:, 1].tolist()
    test_preds =  molbert_clf.predict_proba(embeddings[test_fold_mask])[:, 1].tolist()
    predictions = pd.DataFrame({
        "ID": validation_fold["ID"].tolist() + test_fold["ID"].tolist(),
        "cls_prb": val_preds + test_preds,
        "pKi": np.nan,
        "pIC50": np.nan
    })

    predictions.to_csv(predictions_out_path, index=False)
    print(f"Wrote predictions to {predictions_out_path}.")

    with open(model_out_path, "wb") as f:
        pickle.dump(molbert_clf, f)

    print(f"Saved model at {model_out_path}.")

    return model_out_path