#!/bin/bash
#SBATCH --job-name=dragn_experiment
#SBATCH --account=kmpardo_1874
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --gpus=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

source /home1/lkanduku/.bashrc
conda activate /home1/lkanduku/miniconda3/envs/torch/
set -euo pipefail

DRAGN_EXPERIMENT_DIR="$SLURM_SUBMIT_DIR"
DRAGN_CONFIG_PATH="$DRAGN_EXPERIMENT_DIR/config.yaml"
DRAGN_REPO_ROOT="$(git -C "$DRAGN_EXPERIMENT_DIR" rev-parse --show-toplevel)"

if [[ ! -f "$DRAGN_REPO_ROOT/train.py" ]]; then
    echo "ERROR: train.py not found at $DRAGN_REPO_ROOT/train.py" >&2
    exit 1
fi
if [[ ! -f "$DRAGN_CONFIG_PATH" ]]; then
    echo "ERROR: config.yaml not found in submission directory $DRAGN_EXPERIMENT_DIR" >&2
    echo "Submit with: cd <experiment-folder> && sbatch run.sbatch" >&2
    exit 1
fi

cd "$DRAGN_REPO_ROOT"
python -c "import numpy, PIL, pydantic, torch, yaml"
srun --unbuffered python -u "$DRAGN_REPO_ROOT/train.py" --config "$DRAGN_CONFIG_PATH"
