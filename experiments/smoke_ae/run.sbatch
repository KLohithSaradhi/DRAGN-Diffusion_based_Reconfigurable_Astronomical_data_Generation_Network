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
#SBATCH --chdir=/home1/lkanduku/DRAGN-Diffusion_based_Reconfigurable_Astronomical_data_Generation_Network
#SBATCH --output=/home1/lkanduku/DRAGN-Diffusion_based_Reconfigurable_Astronomical_data_Generation_Network/experiments/smoke_ae/%x_%j.out
#SBATCH --error=/home1/lkanduku/DRAGN-Diffusion_based_Reconfigurable_Astronomical_data_Generation_Network/experiments/smoke_ae/%x_%j.err

source /home1/lkanduku/.bashrc
conda activate /home1/lkanduku/miniconda3/envs/torch/
set -euo pipefail

DRAGN_REPO_ROOT=/home1/lkanduku/DRAGN-Diffusion_based_Reconfigurable_Astronomical_data_Generation_Network
DRAGN_CONFIG_PATH="$DRAGN_REPO_ROOT/experiments/smoke_ae/config.yaml"
DRAGN_DATA_PATH="$DRAGN_REPO_ROOT/ProcessedData"

if [[ ! -f "$DRAGN_REPO_ROOT/train.py" ]]; then
    echo "ERROR: train.py not found at $DRAGN_REPO_ROOT/train.py" >&2
    exit 1
fi
if [[ ! -f "$DRAGN_CONFIG_PATH" ]]; then
    echo "ERROR: experiment config not found at $DRAGN_CONFIG_PATH" >&2
    exit 1
fi
if [[ ! -d "$DRAGN_DATA_PATH" ]]; then
    echo "ERROR: ProcessedData not found at $DRAGN_DATA_PATH" >&2
    exit 1
fi

python -c "import numpy, PIL, pydantic, torch, yaml"
python "$DRAGN_REPO_ROOT/train.py" --config "$DRAGN_CONFIG_PATH"
