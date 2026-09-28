#!/bin/bash
#SBATCH --job-name=dragn_production
#SBATCH --account=kmpardo_1874
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source /home1/lkanduku/.bashrc
conda activate /home1/lkanduku/miniconda3/envs/torch/
set -euo pipefail

if [[ -z "${DRAGN_CONFIG:-}" ]]; then
  echo "Submit with DRAGN_CONFIG=<repo-relative-yaml>" >&2
  exit 2
fi
DRAGN_REPO_ROOT="$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)"
cd "$DRAGN_REPO_ROOT"
python -c "import numpy, PIL, pydantic, torch, yaml"
if [[ "$DRAGN_CONFIG" == experiments/data_generation/* ]]; then
  srun --unbuffered python -u -m dragn.classification.generate_dataset --config "$DRAGN_CONFIG"
else
  srun --unbuffered python -u train.py --config "$DRAGN_CONFIG"
fi

