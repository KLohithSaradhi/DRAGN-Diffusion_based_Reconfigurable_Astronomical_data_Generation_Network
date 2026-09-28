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

The benchmark recognizes two instruments (`SDSS`, `SUBARU`) and five objects (`lens`, `spiral`, `ring`, `companion`, `smooth`). It evaluates six experiments:

1. SDSS real images, object classification.
2. SDSS real + DRAGN images, object classification.
3. SUBARU real images, object classification.
4. SUBARU real + DRAGN images, object classification.
5. All real images, instrument classification.
6. All real + DRAGN images, instrument classification.

Every paired real/augmented experiment uses the same real split and classifier seeds. Synthetic images are training-only. Test metrics are always computed exclusively from held-out real images.

### 1. Create the immutable real split

Do this once, before training DRAGN or a classifier:

Manifests created by older revisions are intentionally rejected because they lack group and content hashes. Regenerate the manifest and retrain all benchmark checkpoints after adopting this workflow.

```bash
python -m dragn.classification.prepare_splits \
  --root /path/to/ProcessedData \
  --output benchmark_data/real_splits.csv \
  --group-id-mode stem \
  --seed 42
```

By default, images with the same filename stem are treated as observations of the same physical object and remain in one split. Exact file duplicates and identical normalized pixels are also merged automatically. If filenames are not stable physical-object identifiers, provide an authoritative mapping:

```csv
path,group_id
/path/to/ProcessedData/SDSS/lens/cutout_1.png,catalogue-object-123
/path/to/ProcessedData/SUBARU/lens/cutout_9.png,catalogue-object-123
```

```bash
python -m dragn.classification.prepare_splits \
  --root /path/to/ProcessedData \
  --output benchmark_data/real_splits.csv \
  --group-map /path/to/object_groups.csv \
  --seed 42
```

The expected real-data hierarchy is:

```text
ProcessedData/
  SDSS/
    lens/
    spiral/
    ring/
    companion/
    smooth/
  SUBARU/
    lens/
    spiral/
    ring/
    companion/
    smooth/
```

The manifest approximates a joint 70% train, 15% validation, and 15% test split while keeping every physical-object and duplicate group intact. The benchmark DRAGN configurations consume only rows marked `train` and have `leakage_safe: true`, so malformed or non-grouped manifests are rejected.

Audit the real split before training:

```bash
python -m dragn.classification.audit_leakage \
  --real-manifest benchmark_data/real_splits.csv
```

### 2. Train leakage-safe production DRAGN models

The canonical production chain is:

```text
experiments/production_ae/config.yaml
experiments/production_flow/config.yaml
experiments/production_flow_<object>_<instrument>_full_lora/config.yaml
```

The ten full-LoRA jobs—one for every instrument/object pair—are independent after the base-flow checkpoint exists and can run in parallel. Existing projection/MLP ablation configs are also leakage-safe.

For Slurm, submit the complete production dependency chain from the experiments directory:

```bash
cd experiments
./submit_production_pipeline.sh
```

This submits a fresh AE, a fresh flow backbone, every production LoRA config in parallel, and finally the combined synthetic export. Production configs use `resume: never` to prevent reuse of pre-split checkpoints.

The supplied YAML files use the repository's existing cluster data path. Update `data.root_dir` if the dataset is elsewhere.

### 3. Export a balanced synthetic training set

After all ten adapters finish:

```bash
python -m dragn.classification.generate_dataset \
  --config experiments/data_generation/all_production_dragn.yaml
```

This writes individual PNG files and `benchmark_data/synthetic/manifest.csv`. It generates one synthetic image per real training image within every instrument/object stratum. Export refuses checkpoints whose base, autoencoder, split-manifest hashes, architecture, objective, labels, or raw/EMA selection do not match.

For an experiment-config-driven export, list the LoRA experiment YAMLs under `sources` and run:

```bash
python -m dragn.classification.generate_dataset \
  --config experiments/data_generation/object_sdss_dragn.yaml
```

Each source label, autoencoder checkpoint, base checkpoint, and default adapter checkpoint is derived from its LoRA experiment YAML. A source can optionally override its checkpoint, weight variant, and adapter scale. The canonical classifier configs consume the combined `benchmark_data/synthetic/manifest.csv`; `object_sdss_dragn.yaml` remains available when generating an SDSS-only dataset for a standalone run.

Audit the real split, generated data, and provenance together:

```bash
python -m dragn.classification.audit_leakage \
  --real-manifest benchmark_data/real_splits.csv \
  --synthetic-manifest benchmark_data/synthetic_sdss/manifest.csv
```

Augmented classifier training fails if this provenance is absent, references a different real manifest, has a modified synthetic manifest, or lacks generator checkpoint hashes.

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
