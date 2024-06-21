#!/usr/bin/env python3

import os
import joblib
import concurrent.futures

import pandas as pd

from utils import compute_descriptors


def process_chunk(args):
    idx = args[0]
    chunk = args[1]

    properties = pd.DataFrame(compute_descriptors(chunk['SMILES'], descriptor))

    chunk['pIC50'] = [x * train_std + train_mean for x in regressor.predict(properties)]
    chunk['pKi'] = chunk['pIC50']

    chunk['cls_prob'] = classifier.predict_proba(properties)[:, 1]

    chunk.drop('SMILES', axis=1, inplace=True)

    chunk.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
    print(f'wrote predictions of chunk {idx}')


if __name__ == '__main__':  # important to avoid recursive initiation
    for descriptor in ['physchem', 'morgan_fp']:
        chunk_size = 10000
        enamine_file = 'test.csv'
        models_dir = 'output/models'
        norm_info = pd.read_csv(os.path.join(models_dir, 'normalization_info.csv'))
        train_mean = norm_info.iloc[0]['mean']
        train_std = norm_info.iloc[0]['std']
        classifier = joblib.load(os.path.join(models_dir, f'rf_classification_{descriptor}.joblib'))
        regressor = joblib.load(os.path.join(models_dir, f'rf_regression_{descriptor}.joblib'))
        output_file = f'enamine_predictions_{descriptor}.csv'

        chunks = [process_chunk([i, chunk])
                  for i, chunk in enumerate(pd.read_csv(enamine_file, chunksize=chunk_size))]

        with concurrent.futures.ProcessPoolExecutor(max_workers=300) as executor:
            executor.map(process_chunk, chunks)

