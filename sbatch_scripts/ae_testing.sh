#!/bin/bash

#SBATCH --job-name=test_run
#SBATCH --account=kmpardo_1874
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --gpus=1
#SBATCH --output=./logs/%x_%j.out 
#SBATCH --error=./logs/%x_%j.err

source /home1/lkanduku/.bashrc
conda activate torch
python3 ./sample_ae.py --run_dir ./results/3_32_64_128_128ae --image /home1/lkanduku/SDSS/spiral/000001.jpg