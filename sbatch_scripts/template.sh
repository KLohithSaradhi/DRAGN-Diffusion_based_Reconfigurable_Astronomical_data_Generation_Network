#!/bin/bash

#SBATCH --account=<project_id>
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --gpu-per-task=1

module purge
module load gcc/13.3.0
module load python/3.11.9

python3 script.py