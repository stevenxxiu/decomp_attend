#!/bin/bash
#PBS -q gpupascal
#PBS -l ngpus=1
#PBS -l ncpus=6
#PBS -l mem=16GB
#PBS -l walltime=20:00:00
module load tensorflow/1.0.1-python3.5
cd /short/cp1/sx6361/decomp_attend
/home/563/sx6361/.pyenv/versions/3.6.1/bin/python decomp_attend/main.py '{
    "intra_sent": true, "emb_unknown": 100, "emb_size": 300,
    "emb_proj": 200, "emb_proj_pca": false, "emb_normalize": true,
    "n_intra": [200, 200], "n_intra_bias": 10, "n_attend": [200, 200], "n_compare": [200, 200], "n_classif": [200, 200],
    "dropout_rate": 0.2, "lr": 0.0005, "batch_size": 256, "epoch_size": 400
}'
