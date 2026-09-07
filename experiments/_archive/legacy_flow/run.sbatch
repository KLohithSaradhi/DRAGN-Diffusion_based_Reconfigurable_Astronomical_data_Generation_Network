#!/bin/bash

#SBATCH --job-name=flow_training
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

echo "starting log"
source /home1/lkanduku/.bashrc
echo "starting log"
conda activate torch
echo "starting log"
python3 ./train.py --config /home1/lkanduku/DRAGN-Diffusion_based_Reconfigurable_Astronomical_data_Generation_Network/runs/2_trans_flow.yaml