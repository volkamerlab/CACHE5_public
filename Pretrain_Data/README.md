# How to create the cleaned data for training

### Setup conda

Run

```shell
conda create -n cache_prep python=3.9
conda activate cache_prep
conda install -c conda-forge numpy pandas matplotlib tqdm
```

### Prepare BindingDB

Navigate to the `BindingDB` folder and place the raw data into a `raw` folder. Then run the following code inside the `cache_prep` environment 

```shell
python main_bdb.py
```

### Prepare GLASS

Navigate to the `GLASS` folder nad place the raw data into a `raw` folder. Then run the following code inside the `cache_prep` environment

```shell
python main_glass.py
```

