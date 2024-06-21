#!/usr/bin/env python3
import copy
import os

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# The below ENV declaration was needed for Mac
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch

import wandb

import evaluate
from datasets import Dataset
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    EarlyStoppingCallback, AutoModelForMaskedLM, DataCollatorForLanguageModeling

# assert torch.cuda.is_available()

os.environ["WANDB_LOG_MODEL"] = "end"


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


def class_prob(logits, labels):
    if len(np.unique(labels)) == 1:
        probs_df = pd.DataFrame({'prediction': logits[:, 0].tolist()})
    else:
        probs = torch.nn.functional.softmax(torch.from_numpy(logits), dim=1)
        index_tensor = torch.from_numpy(labels)
        predicted_probs = probs[range(probs.size(0)), index_tensor].tolist()
        probs_df = pd.DataFrame({'class_prediction': labels, 'class_probability': predicted_probs})
    return probs_df


def get_data(data_path):
    df = pd.read_csv(data_path, index_col=0)
    dl_df = df[~df['DataSAIL_10f'].isin(['Fold_8', 'Fold_9'])]
    ensemble_df = df[df['DataSAIL_10f'].isin(['Fold_8', 'Fold_9'])]
    return dl_df, ensemble_df


def get_splits(cv, continue_train, fold_idx, num_folds, df, fold_col, smiles_col, class_col):
    if continue_train:
        train_idx = df.sample(frac=0.8).index
        train_smiles = df[train_idx].to_list()
        val_smiles = df.drop(train_idx.to_list()).to_list()
        train_targets = val_targets = test_smiles = test_targets = None
    else:
        if isinstance(df, list):
            cv_data = df[0]
            test_data = df[1]
        else:
            cv_data = df
            test_data = None
        if cv:
            test_fold = fold_idx
            val_fold = fold_idx + 1 if fold_idx != num_folds - 1 else 0
            not_train_folds = [f'Fold_{test_fold}', f'Fold_{val_fold}']

            test_idx = cv_data[cv_data[fold_col] == f'Fold_{test_fold}'].index
            test_smiles = cv_data.loc[test_idx][smiles_col].to_list()
            test_targets = cv_data.loc[test_idx][class_col].to_list()
        else:
            val_fold = num_folds - 1
            not_train_folds = [f'Fold_{val_fold}']

            test_smiles = test_data[smiles_col].to_list()
            test_targets = test_data[class_col].to_list()

        val_idx = cv_data[cv_data[fold_col] == f'Fold_{val_fold}'].index
        val_smiles = cv_data.loc[val_idx][smiles_col].to_list()
        val_targets = cv_data.loc[val_idx][class_col].to_list()

        train_idx = cv_data[~cv_data[fold_col].isin(not_train_folds)].index
        train_smiles = cv_data.loc[train_idx][smiles_col].to_list()
        train_targets = cv_data.loc[train_idx][class_col].to_list()
    return train_smiles, train_targets, val_smiles, val_targets, test_smiles, test_targets


def tokenize_data(tokenizer,
                  tokenizer_config,
                  train_smiles,
                  val_smiles,
                  train_targets=None,
                  val_targets=None,
                  test_smiles=None,
                  test_targets=None,
                  continue_train=False,
                  frozen=False):
    def tokenize_input(examples):
        return tokenizer(examples["text"],
                         **tokenizer_config)

    if frozen:
        tokenized_train = tokenizer(train_smiles, **tokenizer_config)
        tokenized_val = tokenizer(val_smiles, **tokenizer_config)
        return tokenized_train, tokenized_val
    elif continue_train:
        train_df = pd.DataFrame(train_smiles)
        val_df = pd.DataFrame(val_smiles)

        train_dataset = Dataset.from_pandas(train_df.rename(columns={0: "text"}))
        val_dataset = Dataset.from_pandas(val_df.rename(columns={0: "text"}))

        tokenized_train = train_dataset.map(tokenize_input, batched=True)
        tokenized_val = val_dataset.map(tokenize_input, batched=True)
        return tokenized_train, tokenized_val, None
    else:
        train_ds = Dataset.from_list(
            [{'label': train_targets[i], 'text': train_smiles[i]} for i in range(len(train_targets))])
        val_ds = Dataset.from_list([{'label': val_targets[i], 'text': val_smiles[i]} for i in range(len(val_targets))])
        test_ds = Dataset.from_list(
            [{'label': test_targets[i], 'text': test_smiles[i]} for i in range(len(test_targets))])

        tokenized_train = train_ds.map(tokenize_input, batched=True)
        tokenized_val = val_ds.map(tokenize_input, batched=True)
        tokenized_test = test_ds.map(tokenize_input, batched=True)
        return tokenized_train, tokenized_val, tokenized_test


def frozen_tuning(model, tokenizer, tokenizer_config, train_smiles, train_targets, test_smiles, test_targets, ft,
                  performance):
    train_inputs, test_inputs = tokenize_data(tokenizer=tokenizer,
                                              tokenizer_config=tokenizer_config,
                                              train_smiles=train_smiles,
                                              val_smiles=test_smiles,
                                              frozen=True)
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
                continue_train,
                training_config,
                train_smiles,
                train_targets,
                val_smiles,
                val_targets,
                test_smiles,
                test_targets,
                num_labels,
                prefix,
                ):
    if continue_train:
        tokenized_train_ds, tokenized_val_ds, tokenized_test_ds = tokenize_data(tokenizer=tokenizer,
                                                                                tokenizer_config=tokenizer_config,
                                                                                train_smiles=train_smiles,
                                                                                val_smiles=val_smiles,
                                                                                continue_train=True
                                                                                )
    else:
        tokenized_train_ds, tokenized_val_ds, tokenized_test_ds = tokenize_data(tokenizer=tokenizer,
                                                                                tokenizer_config=tokenizer_config,
                                                                                train_smiles=train_smiles,
                                                                                val_smiles=val_smiles,
                                                                                test_smiles=test_smiles,
                                                                                train_targets=train_targets,
                                                                                val_targets=val_targets,
                                                                                test_targets=test_targets,
                                                                                )

    if continue_train:
        run_name = f'{prefix}_molformer'
    else:
        run_name = f"fold_{fold_idx}"

    training_args = TrainingArguments(output_dir=f"{prefix}_molformer_cv",
                                      run_name=run_name,
                                      **training_config
                                      )

    if continue_train:
        tokenizer.pad_token = '[PAD]'  # tokenizer.eos_token
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_ds,
            eval_dataset=tokenized_val_ds,
            data_collator=data_collator
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_ds,
            eval_dataset=tokenized_val_ds,
            compute_metrics=compute_metrics_regression if num_labels == 1 else compute_metrics_classification,
        )
    trainer.train()
    if not continue_train:
        logits, _, pred_metrics = trainer.predict(tokenized_test_ds)
        test_pred_prob = class_prob(logits, labels=np.argmax(logits, axis=1))
    else:
        pred_metrics = test_pred_prob = None

    # Take care of distributed/parallel training
    model_to_save = trainer.model.module if hasattr(trainer.model,
                                                    'module') else trainer.model
    model_to_save.save_pretrained(f'molformer_models/{prefix}/Fold_{fold_idx}')
    return pred_metrics, test_pred_prob


def cross_validation(model_name,
                     model_frozen,
                     training_config,
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
                     cv=True,
                     continue_train=False):
    performance = {}
    if ml_dict is not None:
        for ml in ml_dict.keys():
            performance[ml] = []
    if tune:
        performance = []

    for fold_idx in range(num_folds):
        train_smiles, train_targets, val_smiles, val_targets, test_smiles, test_targets = get_splits(cv,
                                                                                                     continue_train,
                                                                                                     fold_idx,
                                                                                                     num_folds,
                                                                                                     df,
                                                                                                     fold_col,
                                                                                                     smiles_col,
                                                                                                     class_col
                                                                                                     )
        if tune:
            if continue_train:
                model_tune = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True)
            else:
                model_tune = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                                trust_remote_code=True,
                                                                                num_labels=num_labels,
                                                                                device_map="auto"
                                                                                )
            pred_metrics, test_probs = fine_tuning(fold_idx=fold_idx,
                                                   model=model_tune,
                                                   tokenizer=tokenizer,
                                                   tokenizer_config=tokenizer_config,
                                                   continue_train=continue_train,
                                                   training_config=training_config,
                                                   train_smiles=train_smiles,
                                                   train_targets=train_targets,
                                                   val_smiles=val_smiles,
                                                   val_targets=val_targets,
                                                   test_smiles=test_smiles,
                                                   test_targets=test_targets,
                                                   num_labels=num_labels,
                                                   prefix=prefix)
            performance.append(pred_metrics)
            wandb.finish()

            if not continue_train and not cv:
                test_probs['ID'] = df[1]['ID'].to_list()  # .loc[test_idx]['ID'].to_list()
                # train_probs['ID'] = df[1]['ID'].to_list()  # .loc[train_idx]['ID'].to_list()

                test_probs.to_csv(f'{fold_idx}_test_probs.csv')
                # train_probs.to_csv(f'{fold_idx}_train_probs.csv')
        if ml_dict is not None:
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
        if not cv:
            break
    if not continue_train:
        performance_df = pd.DataFrame(performance)
        performance_df.to_csv(f'{prefix}_molformer_results.csv')


if __name__ == '__main__':
    model_name = "ibm/MoLFormer-XL-both-10pct"
    tune = True
    further_train = False
    with_no_trunc = False
    classification = True
    cv = False
    DA_frac = 0.003
    epochs = 1

    glass_path = 'GLASS.tsv'
    glass_data = pd.read_csv(glass_path, sep='\t')
    glass_smiles = glass_data[glass_data['Molecular Weight'] <= 700]['Canonical SMILES'].sample(frac=DA_frac)

    cache_path = '../../Reference Data/20240430_MCHR1_splitted_RJ.csv'
    dl_data, ensemble_data = get_data(cache_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # specifying padding length is required.
    # I assume this is because MolFormer uses rotary PE where eos token is not defined.
    tokenizer_config = {'padding': 'max_length',
                        'truncation': True,
                        'max_length': 700,  # tokenizer.model_max_length,
                        'return_tensors': "pt"}
    model_frozen = AutoModel.from_pretrained(model_name, deterministic_eval=True, trust_remote_code=True)
    if tune:
        out_prefix = 'tuned'
        frozen_ml = None
    elif further_train:
        out_prefix = 'DA'
        frozen_ml = None
    else:
        out_prefix = 'frozen'
        if classification:
            frozen_ml = {'rf': RandomForestClassifier()}  # 'svm': SVC(probability=True),
        else:
            frozen_ml = {'svm': SVR(), 'rf': RandomForestRegressor()}
        if with_no_trunc:
            out_prefix = 'frozen_no_trunc'
            tokenizer_config = {'padding': True,
                                'return_tensors': "pt"}

    if not further_train:
        if classification:
            n_labels = 2
            out_prefix = f'classification_{out_prefix}'
            pred_col = 'class'
        else:
            n_labels = 1
            out_prefix = f'regression_{out_prefix}'
            pred_col = 'acvalue_uM'
    else:
        n_labels = None
        pred_col = None

    if cv:
        cache_data = dl_data
    else:
        cache_data = [dl_data, ensemble_data]
        out_prefix = f'final_{out_prefix}'

    trainer_config = {'evaluation_strategy': "epoch",
                      'logging_strategy': "epoch",
                      'save_strategy': 'epoch',
                      'num_train_epochs': epochs,
                      'per_device_train_batch_size': 64,
                      'per_device_eval_batch_size': 16,
                      # bf16 is not supported for Mac
                      # 'bf16': True,
                      'report_to': "wandb",
                      'load_best_model_at_end': True}

    os.environ["WANDB_PROJECT"] = f"cache5_molformer_{out_prefix}"
    cross_validation(model_name=model_name,
                     model_frozen=model_frozen,
                     training_config=trainer_config,
                     tokenizer=tokenizer,
                     tokenizer_config=tokenizer_config,
                     df=cache_data if not further_train else glass_smiles,
                     prefix=out_prefix,
                     num_labels=n_labels,
                     class_col=pred_col,
                     ml_dict=frozen_ml,
                     tune=tune,
                     num_folds=8,
                     cv=False if further_train else cv,
                     continue_train=further_train)
