#!/bin/bash

#SBATCH --job-name=lora_spiral
#SBATCH --account=kmpardo_1874
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --output=./logs/%x_%j.out 
#SBATCH --error=./logs/%x_%j.err
####SBATCH --constraint="a100"

source /home1/lkanduku/.bashrc
conda activate
conda activate /home1/lkanduku/miniconda3/envs/torch/
echo $CONDA_PREFIX
python train.py --config ./runs/big_model_bb_lora_spiral.yaml