#!/bin/bash
#PBS -q gpupascal
#PBS -l ngpus=1
#PBS -l ncpus=6
#PBS -l mem=16GB
module load tensorflow/1.0.1-python3.5
cd /short/cp1/sx6361/decomp_attend
/home/563/sx6361/.pyenv/versions/3.6.1/bin/python decomp_attend/main.py
