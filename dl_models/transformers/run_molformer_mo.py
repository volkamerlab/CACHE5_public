#!/usr/bin/env python3
import os
import pathlib

import evaluate
import numpy as np
import pandas as pd
# The below ENV declaration was needed for Mac
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
from datasets import Dataset
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVC, SVR
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

import wandb

from multi_objective import MolformerForMultiobjective

assert torch.cuda.is_available()


def compute_metrics_classification(eval_pred, metric='roc_auc'):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metric_func = evaluate.load(metric)
    return metric_func.compute(prediction_scores=predictions, references=labels)


def compute_metrics_regression(eval_pred):
    predictions, labels = eval_pred
    rmse = mean_squared_error(labels, predictions, squared=False)
    mae = mean_absolute_error(labels, predictions)
    return {"rmse": rmse, 'mae': mae}


def compute_metrics_multi_objective(eval_pred, classification_metric="roc_auc"):
    predictions, labels = eval_pred
    regression_preds, classification_preds = predictions
    regression_labels, classification_labels = labels

    classification_metrics = compute_metrics_classification((classification_preds, classification_labels), classification_metric)
    regression_metrics = compute_metrics_regression((regression_preds, regression_labels))
    return {**classification_metrics, **regression_metrics}


def class_prob(logits, labels):
    if len(np.unique(labels)) == 1:
        probs_df = pd.DataFrame({'prediction': logits[:, 0].tolist()})
    else:
        probs = torch.nn.functional.softmax(torch.from_numpy(logits), dim=1)
        index_tensor = torch.from_numpy(labels)
        predicted_probs = probs[range(probs.size(0)), index_tensor].tolist()
        probs_df = pd.DataFrame({'class_prediction': labels, 'class_probability': predicted_probs})
    return probs_df


def full_predictions_df(y_hat, logits):
    predicted_labels = np.argmax(logits, axis=1)
    probs = torch.nn.functional.softmax(torch.from_numpy(logits), dim=1)
    index_tensor = torch.from_numpy(predicted_labels)
    predicted_probs = probs[range(probs.size(0)), index_tensor].tolist()
    return pd.DataFrame({
        'class_prediction': predicted_labels.tolist(),
        'class_probability': predicted_probs,
        'regression_prediction': y_hat[:, 0].tolist()}
    )


def get_data(data_path):
    df = pd.read_csv(data_path, index_col=0)
    df = df[~df['DataSAIL_10f'].isin(['Fold_8', 'Fold_9'])]
    return df


def frozen_tuning(model, tokenizer, tokenizer_config, train_smiles, train_targets, test_smiles, test_targets, ft,
                  performance):
    train_inputs = tokenizer(train_smiles, **tokenizer_config)
    test_inputs = tokenizer(test_smiles, **tokenizer_config)
    with torch.no_grad():
        train_outputs = model(**train_inputs)
        test_outputs = model(**test_inputs)
    train_rep = train_outputs.pooler_output
    test_rep = test_outputs.pooler_output

    for ft_method, clf in ft.items():
        clf.fit(train_rep, train_targets)
        preds = clf.predict(test_rep)
        performance[ft_method].append(metrics.roc_auc_score(test_targets, preds))


def fine_tuning(fold_idx,
                model,
                tokenizer,
                tokenizer_config,
                train_ds,
                val_ds,
                test_ds,
                prefix,
                num_labels,
                epoch=10,
                ):

    def tokenize_input(examples):
        return tokenizer(examples["text"],
                         **tokenizer_config)

    tokenized_train_ds = train_ds.map(tokenize_input, batched=True)
    tokenized_val_ds = val_ds.map(tokenize_input, batched=True)
    tokenized_test_ds = test_ds.map(tokenize_input, batched=True)

    checkpoint_dir = os.path.join(os.getenv("CHECKPOINT_DIR", "."), f"{prefix}_molformer_cv_checkpoints/")
    training_args = TrainingArguments(output_dir=checkpoint_dir,
                                      evaluation_strategy="epoch",
                                      logging_strategy="epoch",
                                      save_strategy='epoch',
                                      do_train=True,
                                      do_eval=True,
                                      num_train_epochs=epoch,
                                      report_to="wandb",
                                      run_name=f"fold_{fold_idx}",
                                      load_best_model_at_end=True,
                                      save_total_limit=1,
                                      )
    if num_labels == 1:
        metrics_fct = compute_metrics_regression
    elif num_labels == 2:
        metrics_fct = compute_metrics_classification
    else:
        metrics_fct = compute_metrics_multi_objective

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_ds,
        eval_dataset=tokenized_val_ds,
        compute_metrics=metrics_fct,
    )
    trainer.train()
    logits, _, pred_metrics = trainer.predict(tokenized_test_ds)

    # MultiObjective training returns regression and classification predictions
    if num_labels == -1:
        y_hat, logits = logits
        test_pred_prob = full_predictions_df(y_hat, logits)
    else:
        test_pred_prob = class_prob(logits, labels=np.argmax(logits, axis=1))

    logits, _, pred_metrics = trainer.predict(tokenized_train_ds)
    if num_labels == -1:  # MultiObjective training returns regression and classification predictions
        y_hat, logits = logits
        train_pred_prob = full_predictions_df(y_hat, logits)
    else:
        train_pred_prob = class_prob(logits, labels=np.argmax(logits, axis=1))

    # Take care of distributed/parallel training
    model_to_save = trainer.model.module if hasattr(trainer.model,
                                                    'module') else trainer.model
    save_dir = os.path.join(os.getenv("CHECKPOINT_DIR", "."), f'molformer_cv_models/{prefix}/cv_{fold_idx}')
    model_to_save.save_pretrained(save_dir)
    return pred_metrics, test_pred_prob, train_pred_prob


def cross_validation(model_name,
                     model_frozen,
                     tokenizer,
                     tokenizer_config,
                     df,
                     prefix,
                     num_labels,
                     num_folds=8,
                     fold_col='DataSAIL_10f',
                     smiles_col='smiles',
                     class_col='class',
                     ml_dict=None,
                     tune=True,
                     num_epoch=10,
                     cv=True):
    multi_objective_model = True if num_labels == -1 else False  # Hack

    results_dir = os.path.join(os.getenv('RESULTS_DIR', '.'), prefix)
    os.mkdir(results_dir)

    performance = {}
    if ml_dict is not None:
        for ml in ml_dict.keys():
            performance[ml] = []
    if tune:
        performance = []

    test_fold = num_folds - 1
    val_fold = num_folds - 2
    not_train_folds = [f'Fold_{test_fold}', f'Fold_{val_fold}']
    if not cv:
        num_folds = 1
    for fold_idx in range(num_folds):
        if cv:
            test_fold = fold_idx
            val_fold = fold_idx + 1 if fold_idx != num_folds - 1 else 0
            not_train_folds = [f'Fold_{test_fold}', f'Fold_{val_fold}']

        test_idx = df[df[fold_col] == f'Fold_{test_fold}'].index
        val_idx = df[df[fold_col] == f'Fold_{val_fold}'].index
        train_idx = df[~df[fold_col].isin(not_train_folds)].index

        if tune:
            if multi_objective_model:
                assert isinstance(class_col, list), "MultiObjective requires more than one label."

                model_tune = MolformerForMultiobjective.from_pretrained(model_name, classification_loss_weight=0.8)
                # We assume that the class label for the regression is at the first index
                train_ds = Dataset.from_pandas(
                    df.loc[train_idx][[smiles_col, *class_col]].rename(
                        columns={'smiles': 'text', class_col[0]: 'label_regression', class_col[1]: 'label_classification'}
                    ),
                    preserve_index=False)
                val_ds = Dataset.from_pandas(
                    df.loc[val_idx][[smiles_col, *class_col]].rename(
                        columns={'smiles': 'text', class_col[0]: 'label_regression', class_col[1]: 'label_classification'}
                    ),
                    preserve_index=False)
                test_ds = Dataset.from_pandas(
                    df.loc[test_idx][[smiles_col, *class_col]].rename(
                        columns={'smiles': 'text', class_col[0]: 'label_regression', class_col[1]: 'label_classification'}
                    ),
                    preserve_index=False)

            else:
                model_tune = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                                trust_remote_code=True,
                                                                                device_map="auto"
                                                                                )

                train_ds = Dataset.from_pandas(
                    df.loc[train_idx][[smiles_col, class_col]].rename(columns={'smiles': 'text', class_col: 'label'}),
                    preserve_index=False)
                val_ds = Dataset.from_pandas(
                    df.loc[val_idx][[smiles_col, class_col]].rename(columns={'smiles': 'text', class_col: 'label'}),
                    preserve_index=False)
                test_ds = Dataset.from_pandas(
                    df.loc[test_idx][[smiles_col, class_col]].rename(columns={'smiles': 'text', class_col: 'label'}),
                    preserve_index=False)

            pred_metrics, test_probs, train_probs = fine_tuning(fold_idx,
                                                                model_tune,
                                                                tokenizer,
                                                                tokenizer_config,
                                                                train_ds,
                                                                val_ds,
                                                                test_ds,
                                                                prefix,
                                                                num_labels=num_labels,
                                                                epoch=num_epoch)
            performance.append(pred_metrics)
            wandb.finish()

            test_probs['ID'] = df.loc[test_idx]['ID'].to_list()
            train_probs['ID'] = df.loc[train_idx]['ID'].to_list()

            test_probs.to_csv(f'{results_dir}/{fold_idx}_test_probs.csv')
            train_probs.to_csv(f'{results_dir}/{fold_idx}_train_probs.csv')
        if ml_dict is not None:
            test_smiles = df.loc[test_idx][smiles_col].to_list()
            test_targets = df.loc[test_idx][class_col].to_list()

            val_smiles = df.loc[val_idx][smiles_col].to_list()
            val_targets = df.loc[val_idx][class_col].to_list()

            train_smiles = df.loc[train_idx][smiles_col].to_list()
            train_targets = df.loc[train_idx][class_col].to_list()

            frozen_tuning(model_frozen,
                          tokenizer,
                          tokenizer_config,
                          train_smiles,
                          train_targets,
                          test_smiles,
                          test_targets,
                          ml_dict,
                          performance,
                          )
        print(f'fold_{fold_idx}', performance)
    performance_df = pd.DataFrame(performance)
    performance_df.to_csv(f'{results_dir}/{prefix}_molformer_results.csv')

    best_roc_idx = performance_df["test_roc_auc"].idxmax()
    best_ckpt = os.path.join(os.getenv("CHECKPOINT_DIR", "."), f'molformer_cv_models/{prefix}/cv_{best_roc_idx}')
    return best_ckpt


def train_molformer(
        checkpoint_dir=None,
        results_dir=None
):
    # `cross_validation` looks for these environment variables during training
    if checkpoint_dir is not None:
        os.environ["CHECKPOINT_DIR"] = checkpoint_dir
    if results_dir is not None:
        os.environ["RESULTS_DIR"] = results_dir

    tune = True
    with_no_trunc = False
    classification = False
    multi_objective = True
    cv = True

    mo_classification_loss_weight = 0.8
    here = pathlib.Path(__file__).parent

    d_path = here.parent / 'Reference Data' / '20240430_MCHR1_splitted_RJ.csv'
    data = get_data(str(d_path))

    model_name = "ibm/MoLFormer-XL-both-10pct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer_config = {'padding': 'max_length',
                        'truncation': True,
                        'max_length': tokenizer.model_max_length,
                        'return_tensors': "pt"}
    model_frozen = AutoModel.from_pretrained(model_name, deterministic_eval=True, trust_remote_code=True)
    if tune:
        out_prefix = 'tuned'
        frozen_ml = None
    else:
        out_prefix = 'frozen'
        if classification:
            frozen_ml = {'svm': SVC(probability=True), 'rf': RandomForestClassifier()}
        else:
            frozen_ml = {'svm': SVR(), 'rf': RandomForestRegressor()}
        if with_no_trunc:
            out_prefix = 'frozen_no_trunc'
            tokenizer_config = {'padding': True,
                                'return_tensors': "pt"}

    if classification:
        num_labels = 2
        out_prefix = f'classification_{out_prefix}'
        pred_col = 'class'
    elif multi_objective:
        num_labels = -1
        out_prefix = f'multiobjective_{mo_classification_loss_weight}_{out_prefix}'
        pred_col = ['acvalue_uM', 'class']  # Regression at the first index, classification on the second
    else:
        num_labels = 1
        out_prefix = f'regression_{out_prefix}'
        pred_col = 'acvalue_uM'

    if not cv:
        out_prefix = f'final_{out_prefix}'
    os.environ["WANDB_PROJECT"] = f"cache5_molformer_{out_prefix}"
    os.environ["WANDB_LOG_MODEL"] = "end"
    cross_validation(model_name,
                     model_frozen,
                     tokenizer,
                     tokenizer_config,
                     data,
                     out_prefix,
                     num_labels,
                     class_col=pred_col,
                     ml_dict=frozen_ml,
                     num_epoch=10,
                     tune=True,
                     num_folds=8,
                     cv=cv)


if __name__ == '__main__':
    train_molformer()
