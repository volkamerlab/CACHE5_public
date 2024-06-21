# Description

The model itself is based on GIN model implemented at https://github.com/yuyangw/MolCLR.
Original idea was to start with pretrained weights from MolCLR paper (because this pretraining might have helped us to obtain better generalization even with small train dataset) and finetune this model, but I simply forgot to load them, as a result I trained the model 'from scratch'.

The loss is a combination of MSE loss, ranking loss and BCE loss, which should allow us to do multi-task learning. The idea of ranking loss originates from this code https://github.com/jiaxianyan/MBP/blob/main/MBP/losses/losses.py and corresponding MBP paper.

## Training

This model was trained in kaggle notebooks with P100 accelerator. 
Currently the notebooks are provided 'as is'. 

- `cache5-molclr-baseline-notebook.ipynb` contains code used for inference.
I'll probably clean this code in the future, since the notebook was cloned from the training one, and the lightning model contains methods/code which was used for training and not required for inference.

Note that this notebook has some kaggle-specific code related to W&B logging. Also, apparently, it was run with outdated version of the data:
https://github.com/volkamerlab/CACHE5/blob/449679f7ce5fadd6ab7ee0389469e5a6a8454bca/Reference%20Data/20240430_MCHR1_splitted_RJ.csv - if you'll run it with the most recent version of this file, the results will be slightly different.

Best weights obtained during training are stored at `checkpoints/epoch=64-step=195-val_loss=4.5169-classif_roc_auc_0=0.6545.ckpt`. I also used them to run interence on enamine dataset.

Currently the path to the training data is given as it was in my kaggle notebook: I've upload the csv file as a separate dataset, load it from this dataset.

During training, 'Fold_7' was used for validation, folds from 'Fold_0' to 'Fold_6' were used for training. I didn't have enough time to run this with other train/val splits.


## Inference on enamine dataset

The data was splitted to 10 parts, 5_000_000 records each (with the remainder 10th part with less than 5_000_000 records), and processed with the previously trained model.

- `cache5-molclr-inference-enamine-5678.ipynb` has the code which was used for inference and processes several parts of this file.

Later, the computed predictions for all 10 parts were merged together.

This notebook also has kaggle-specific code related to W&B logging. 

# Requirements

The code was mainly run at kaggle environment, the latest available versions of packages were used (the notebooks loads only necessary additional packages not present by default in the 1st cell of the notebooks). `requirements.txt` file contains the particular versions of the main packages.

The packeges not listed in the requirements were the one installed at the latest version of kaggle's docker image, which is described here: https://github.com/Kaggle/docker-python