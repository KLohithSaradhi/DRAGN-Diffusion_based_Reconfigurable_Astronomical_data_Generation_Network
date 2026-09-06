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

source /home1/lkanduku/.bashrc
conda activate torch
python3 ./conditioned_sampler.py --base_config /home1/lkanduku/DRAGN-Diffusion_based_Reconfigurable_Astronomical_data_Generation_Network/runs/mnist_flow.yaml --bb_weights /home1/lkanduku/DRAGN-Diffusion_based_Reconfigurable_Astronomical_data_Generation_Network/results/mnist_flow_cosine/flow_bb_weights.pth --obj_config /home1/lkanduku/DRAGN-Diffusion_based_Reconfigurable_Astronomical_data_Generation_Network/runs/mnist_flow_digit_5.yaml --inst_config /home1/lkanduku/DRAGN-Diffusion_based_Reconfigurable_Astronomical_data_Generation_Network/runs/mnist_flow_instrument_lora.yaml --num_samples 16 --num_steps 100 --degrade_steps 60