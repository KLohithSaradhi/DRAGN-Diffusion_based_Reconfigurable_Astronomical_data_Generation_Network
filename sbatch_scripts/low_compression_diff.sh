#!/bin/bash

#SBATCH --job-name=test_run
#SBATCH --account=kmpardo_1874
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --output=./logs/%x_%j.out 
#SBATCH --error=./logs/%x_%j.err

source /home1/lkanduku/.bashrc
conda activate torch
python3 ./train.py --config /home1/lkanduku/DRAGN-Diffusion_based_Reconfigurable_Astronomical_data_Generation_Network/runs/low_compression_diff.yaml