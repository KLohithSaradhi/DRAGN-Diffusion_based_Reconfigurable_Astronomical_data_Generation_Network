#!/bin/bash
#SBATCH --job-name=dragn_cls
#SBATCH --account=kmpardo_1874
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source /home1/lkanduku/.bashrc
conda activate /home1/lkanduku/miniconda3/envs/torch/
set -euo pipefail

DRAGN_REPO_ROOT="$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)"
cd "$DRAGN_REPO_ROOT"
mkdir -p experiments/classification/logs
python -c "import numpy, PIL, pydantic, torch, torchvision, yaml"
srun --unbuffered python -u -m dragn.classification.run_suite --experiments experiments/classification --seeds 41 42 43
