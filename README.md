# DRAGN

DRAGN is a latent astronomical-image generator and a leakage-safe ResNet18 benchmark for measuring whether its synthetic images improve downstream classification.

## Generative pipeline

The v2 pipeline is YAML-driven and has three training stages:

1. Train `AutoencoderKL` to compress images into scaled latents.
2. Train an unconditional latent DiT with flow matching or DDPM.
3. Freeze the DiT and train an instrument/object-specific LoRA adapter.

Run any v2 training or comparison-inference configuration with:

```bash
python train.py --config path/to/config.yaml
```

## Classification benchmark

The benchmark recognizes exactly two instruments (`SDSS`, `SUBARU`) and two objects (`lens`, `spiral`). It evaluates six experiments:

1. SDSS real images, object classification.
2. SDSS real + DRAGN images, object classification.
3. SUBARU real images, object classification.
4. SUBARU real + DRAGN images, object classification.
5. All real images, instrument classification.
6. All real + DRAGN images, instrument classification.

Every paired real/augmented experiment uses the same real split and classifier seeds. Synthetic images are training-only. Test metrics are always computed exclusively from held-out real images.

### 1. Create the immutable real split

Do this once, before training DRAGN or a classifier:

```bash
python -m dragn.classification.prepare_splits \
  --root /path/to/ProcessedData \
  --output benchmark_data/real_splits.csv \
  --seed 42
```

The expected real-data hierarchy is:

```text
ProcessedData/
  SDSS/
    lens/
    spiral/
  SUBARU/
    lens/
    spiral/
```

The manifest is jointly stratified by instrument and object into 70% train, 15% validation, and 15% test. The benchmark DRAGN configurations consume only rows marked `train`.

### 2. Train leakage-safe DRAGN models

Run these configurations in order:

```text
experiments/benchmark_dragn/autoencoder/config.yaml
experiments/benchmark_dragn/base_flow/config.yaml
experiments/benchmark_dragn/lora_sdss_lens/config.yaml
experiments/benchmark_dragn/lora_sdss_spiral/config.yaml
experiments/benchmark_dragn/lora_subaru_lens/config.yaml
experiments/benchmark_dragn/lora_subaru_spiral/config.yaml
```

The four LoRA jobs are independent after the base-flow checkpoint exists and can run in parallel. They all use the approved `full_lora` preset.

For Slurm, submit a stage from the benchmark experiment directory with:

```bash
cd experiments/benchmark_dragn
sbatch --export=ALL,DRAGN_CONFIG=experiments/benchmark_dragn/autoencoder/config.yaml \
  run.sbatch
```

After preparing the manifest, `submit_pipeline.sh` submits the complete dependency chain: AE, base, four parallel LoRAs, then synthetic export.

The supplied YAML files use the repository's existing cluster data path. Update `data.root_dir` if the dataset is elsewhere.

### 3. Export a balanced synthetic training set

After all four adapters finish:

```bash
python -m dragn.classification.export_synthetic \
  --config experiments/benchmark_dragn/export/config.yaml
```

This writes individual PNG files and `benchmark_data/synthetic/manifest.csv`. It generates one synthetic image per real training image within every instrument/object stratum. Export refuses checkpoints whose base, autoencoder, split-manifest hashes, architecture, objective, labels, or raw/EMA selection do not match.

### 4. Run all six classifier experiments

Run three paired seeds locally:

```bash
python -m dragn.classification.run_suite \
  --experiments experiments/classification \
  --seeds 41 42 43
```

Or submit `experiments/classification/run.sbatch` from its directory.

Each run uses a standard ImageNet-pretrained, fully fine-tuned torchvision ResNet18 at 224×224 resolution. Training uses horizontal/vertical flips and arbitrary rotations, AdamW, cosine learning-rate decay, and early stopping on validation macro-F1.

Run artifacts include:

- Resolved configuration and copies of the exact manifests.
- Best checkpoint and epoch history.
- Real-only test predictions.
- Accuracy, balanced accuracy, macro-F1, ROC-AUC, and per-class metrics.
- Confusion matrix as JSON and CSV.
- `results.csv` with every seed, `summary.csv` with means and standard deviations, and `paired_deltas.csv` with augmented-versus-real macro-F1 changes.

Enable W&B by changing `logging.wandb` in the relevant YAML files.

## Tests

```bash
python -m unittest discover -s tests -v
```

Dependencies are listed in `requirements.txt`. ImageNet weights may be downloaded by torchvision on the first classifier run.
