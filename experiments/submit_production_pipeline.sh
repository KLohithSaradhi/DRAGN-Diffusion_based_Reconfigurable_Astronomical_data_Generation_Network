#!/bin/bash
set -euo pipefail

if [[ ! -f ../benchmark_data/real_splits.csv ]]; then
  echo "Missing ../benchmark_data/real_splits.csv; prepare and audit the split first." >&2
  exit 2
fi

ae_job="$(sbatch --parsable --export=ALL,DRAGN_CONFIG=experiments/production_ae/config.yaml production_run.sbatch)"
base_job="$(sbatch --parsable --dependency="afterok:$ae_job" --export=ALL,DRAGN_CONFIG=experiments/production_flow/config.yaml production_run.sbatch)"

lora_jobs=()
for config in production_flow_*_lora/config.yaml; do
  lora_jobs+=("$(sbatch --parsable --dependency="afterok:$base_job" --export="ALL,DRAGN_CONFIG=experiments/$config" production_run.sbatch)")
done

lora_dependency="$(IFS=:; echo "${lora_jobs[*]}")"
export_job="$(sbatch --parsable --dependency="afterok:$lora_dependency" --export=ALL,DRAGN_CONFIG=experiments/data_generation/all_production_dragn.yaml production_run.sbatch)"

echo "autoencoder_job=$ae_job"
echo "base_flow_job=$base_job"
echo "lora_jobs=${lora_jobs[*]}"
echo "synthetic_export_job=$export_job"

