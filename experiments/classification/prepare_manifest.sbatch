#!/bin/bash
#SBATCH --job-name=dragn_split
#SBATCH --account=kmpardo_1874
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source /home1/lkanduku/.bashrc
conda activate /home1/lkanduku/miniconda3/envs/torch/
set -euo pipefail

DRAGN_REPO_ROOT="$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)"
cd "$DRAGN_REPO_ROOT"
mkdir -p experiments/classification/logs benchmark_data
python -m dragn.classification.prepare_splits \
  --root /home1/lkanduku/DRAGN-Diffusion_based_Reconfigurable_Astronomical_data_Generation_Network/ProcessedData \
  --output benchmark_data/real_splits.csv \
  --seed 42
