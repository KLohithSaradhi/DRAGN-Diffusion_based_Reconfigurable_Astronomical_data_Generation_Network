#!/bin/bash
#SBATCH --job-name=dragn_smoke_ae
#SBATCH --account=kmpardo_1874
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --gpus=1
#SBATCH --output=experiments/smoke_ae/%x_%j.out
#SBATCH --error=experiments/smoke_ae/%x_%j.err

source /home1/lkanduku/.bashrc
conda activate /home1/lkanduku/miniconda3/envs/torch/
python train.py --config experiments/smoke_ae/config.yaml
