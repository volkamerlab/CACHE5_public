import pandas as pd
from rdkit.Chem import MolFromSmiles, Descriptors, rdFingerprintGenerator
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator


def compute_descriptors(smiles, method='PhysChem'):
    mol = MolFromSmiles(smiles)
    if method == 'PhysChem':
        descriptors = [name for name, _ in Descriptors.descList]
        calculator = MolecularDescriptorCalculator(descriptors)
        mol_descriptors = calculator.CalcDescriptors(mol)
    elif method == 'Morgan FPs':
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
        mol_descriptors = fpgen.GetFingerprintAsNumPy(mol)
    else:
        raise NotImplemented
    return mol_descriptors


def get_folds(data: pd.DataFrame, num_folds: int, train_val_only: bool):
    cv_train_val_only = []
    cv_train_val_test = []
    for fold_idx in range(num_folds):
        test_fold = fold_idx
        val_fold = fold_idx + 1 if fold_idx != num_folds - 1 else 0
        test_indices = data[data.DataSAIL_10f == f'Fold_{test_fold}'].index
        val_indices = data[data.DataSAIL_10f == f'Fold_{val_fold}'].index
        train_indices = data[~data.DataSAIL_10f.isin([f'Fold_{test_fold}', f'Fold_{val_fold}'])].index
        cv_train_val_only.append((train_indices, val_indices))
        cv_train_val_test.append((train_indices, val_indices, test_indices))
    if train_val_only:
        return cv_train_val_only
    else:
        return cv_train_val_test
