#!/bin/bash

#SBATCH --job-name=ae_instrument_data
#SBATCH --account=kmpardo_1874
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --gpus=1
#SBATCH --output=./logs/%x_%j.out 
#SBATCH --error=./logs/%x_%j.err
#SBATCH --constraint="a40|a100-80gb"

source /home1/lkanduku/.bashrc
conda activate torch
echo $CONDA_PREFIX
python train.py --config ./runs/processed_no_ae_bb.yaml