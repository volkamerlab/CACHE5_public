import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error

from rdkit.Chem import MolFromSmiles, Descriptors, rdFingerprintGenerator
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator


def compute_descriptors(smiles: list[str], method: str):
    mols = [MolFromSmiles(mol) for mol in smiles]
    if method == 'physchem':
        descriptors = [name for name, _ in Descriptors.descList]
        calculator = MolecularDescriptorCalculator(descriptors)
        mol_descriptors = [calculator.CalcDescriptors(mol)
                           for mol in tqdm(mols, desc=f'calculating {method}')]
    elif method == 'morgan_fp':
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
        mol_descriptors = [fpgen.GetFingerprintAsNumPy(mol)
                           for mol in tqdm(mols, desc=f'calculating {method}')]
    else:
        raise NotImplemented
    return mol_descriptors


def get_cv_splits(cv_df, num_folds, fold_col):
    train_indices = []
    val_indices = []
    for fold_idx in range(num_folds):
        test_fold = fold_idx

        test_idx = cv_df[cv_df[fold_col] == f'Fold_{test_fold}'].index.to_list()
        train_idx = cv_df[cv_df[fold_col] != f'Fold_{test_fold}'].index.to_list()

        train_indices.append(train_idx)
        val_indices.append(test_idx)
    return train_indices, val_indices


def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    data = [(d - mean) / std for d in data]
    return data, mean, std


def unnormalize(data, train_std, train_mean):
    return [(x * train_std) + train_mean for x in data]


def train_and_pred_regression(train_features, train_targets, test_features, test_targets, model_params):
    model = RandomForestRegressor(**model_params)
    train_targets, mean, std = normalize(train_targets)
    model.fit(train_features, train_targets)

    preds = model.predict(test_features)
    preds = unnormalize(preds, std, mean)

    rmse = mean_squared_error(test_targets, preds, squared=False)
    mae = mean_absolute_error(test_targets, preds)
    return model, preds, rmse, mae


def train_and_pred_classification(train_features, train_targets, test_features, test_targets, model_params):
    model = RandomForestClassifier(**model_params)
    model.fit(train_features, train_targets)

    preds = model.predict(test_features)
    probs = model.predict_proba(test_features)[:, 1]

    auroc = roc_auc_score(test_targets, preds)
    return model, probs, auroc


def do_cv(train_features,
          train_targets,
          test_features,
          test_targets,
          classification,
          model_params,
          ):
    performance = []
    for fold_idx in range(len(train_features)):
        fold_train_features = train_features[fold_idx]
        fold_train_targets = train_targets[fold_idx]

        fold_test_features = test_features[fold_idx]
        fold_test_targets = test_targets[fold_idx]

        if classification:
            model, probs, roc = train_and_pred_classification(fold_train_features,
                                                              fold_train_targets,
                                                              fold_test_features,
                                                              fold_test_targets,
                                                              model_params)
            performance.append(roc)
        else:
            model, preds, rmse, mae = train_and_pred_regression(fold_train_features,
                                                                fold_train_targets,
                                                                fold_test_features,
                                                                fold_test_targets,
                                                                model_params)
            performance.append([rmse, mae])
    return performance


def fetch_best(performance_dict, reg_metric=None):
    for featurizer, performance_df in performance_dict.items():
        max_avg = max(performance_df.mean())
        corresponding_idx = performance_df.mean().idxmax().tolist()
        if reg_metric is not None:
            featurizer = f'{featurizer} - {reg_metric}'
        print(f'max average {featurizer}: ', max_avg, '---> Corresponding index:', corresponding_idx)


def plot_sem(plot_dict, descriptor, classification, reg_metric=None, num_folds=8, ylim=None):
    f, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout="tight")
    for idx, key in enumerate(plot_dict.keys()):
        ax = axs[idx]
        df = plot_dict[key]
        sns.pointplot(data=df, errorbar='se', capsize=.3, ax=ax, label='dataSAIL')
        x_min, x_max = ax.get_xlim()
        ax.hlines(max(df.mean()), x_min, x_max, colors='green')
        ax.hlines(min(df.mean()), x_min, x_max, colors='red')
        if ylim is not None:
            ax.set_ylim(ylim)
        if classification:
            ax.set_ylabel('AUROC $\\pm$ SE', fontsize=12)
        else:
            ax.set_ylabel(f'{reg_metric.upper()} $\\pm SE$', fontsize=12)
        ax.set_xlabel('Parameter combination index', fontsize=12)
        ax.set_title(key)
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
    f.suptitle(f'{num_folds}-Fold Grid Search - RF - {descriptor}', fontsize=16)
    plt.show()

