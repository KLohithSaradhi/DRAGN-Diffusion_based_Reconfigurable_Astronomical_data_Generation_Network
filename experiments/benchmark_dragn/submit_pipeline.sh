#!/bin/bash
set -euo pipefail

if [[ ! -f ../../benchmark_data/real_splits.csv ]]; then
  echo "Missing benchmark_data/real_splits.csv; prepare the immutable split first." >&2
  exit 2
fi

ae_job="$(sbatch --parsable --export=ALL,DRAGN_CONFIG=experiments/benchmark_dragn/autoencoder/config.yaml run.sbatch)"
base_job="$(sbatch --parsable --dependency="afterok:$ae_job" --export=ALL,DRAGN_CONFIG=experiments/benchmark_dragn/base_flow/config.yaml run.sbatch)"

lora_jobs=()
for stage in \
  lora_sdss_lens lora_sdss_spiral lora_sdss_ring lora_sdss_companion lora_sdss_smooth \
  lora_subaru_lens lora_subaru_spiral lora_subaru_ring lora_subaru_companion lora_subaru_smooth; do
  lora_jobs+=("$(sbatch --parsable --dependency="afterok:$base_job" --export="ALL,DRAGN_CONFIG=experiments/benchmark_dragn/$stage/config.yaml" run.sbatch)")
done

lora_dependency="$(IFS=:; echo "${lora_jobs[*]}")"
export_job="$(sbatch --parsable --dependency="afterok:$lora_dependency" --export=ALL,DRAGN_CONFIG=experiments/benchmark_dragn/export/config.yaml run.sbatch)"

echo "autoencoder_job=$ae_job"
echo "base_job=$base_job"
echo "lora_jobs=${lora_jobs[*]}"
echo "synthetic_export_job=$export_job"
echo "After export job $export_job succeeds, submit ../classification/run.sbatch from experiments/classification."
