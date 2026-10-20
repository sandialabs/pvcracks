# pvcracks-training

`pvcracks-training` is the optional companion package for retraining
MultiSolSegment, the 3-channel autoencoder, and XGBoost models. It is separated from the main `pvcracks`
distribution so a normal installation remains focused on the documented
analysis, pretrained-inference, XGBoost, and SPICE workflows.

Install both projects from this repository:

```bash
python -m pip install -e .
python -m pip install -e ./training
```

The separation keeps experiment orchestration and training entry points out of
the inference wheel. Model architecture and feature preparation live in
`pvcracks`, shared by training and inference, so checkpoint definitions cannot
drift between two copies. The base training install uses runtime dependencies;
notebook and Ray/HPO tools are opt-in:

```bash
python -m pip install -e "./training[notebooks]"
# For MultiSolSegment Ray/HPO experiments:
python -m pip install -e "./training[hpo,notebooks]"
```

Layout:

```text
training/
  scripts/                 command-line entry points and Slurm submission
  notebooks/               XGBoost and MultiSolSegment training notebooks
  src/pvcracks_training/    training loops, losses, and experiment helpers
  legacy/                  historical one-channel experiment
```

## Current 3-channel autoencoder

The FinalVAE3CH model is `UNetAutoencoder3CHPlus`, a deterministic U-Net
autoencoder with skip connections, separate output heads, and crack deep
supervision. The historical `vae` namespace is retained. The supplied experiment
is ported to `src/pvcracks_training/vae/train_3ch.py`, using the runtime model
definition. It no longer needs `custom_dataset2`, `VAE_functions`, or fixed HPC
paths, and importing it does not start training or change directories.

Run from the repository root:

```bash
python training/scripts/train_3ch_autoencoder.py --crack-dir /data/crack --busbar-dir /data/busbar --dark-dir /data/dark --output-dir outputs/autoencoder_3ch
```

Each directory contains matching 400 x 400 TIFF masks with unique numeric
prefixes such as `12_crack.tif`, `12_busbar.tif`, and `12_dark.tif`. The script
matches prefixes across channels and thresholds each image at its mean, as in
the supplied experiment. It retains the original losses, augmentation, sampler,
AdamW settings, scheduler, early stopping, and five-fold cross validation.
Use `--help` for epochs, folds, batch size, base channels, learning rate, seed,
workers, and device settings. Workers default to zero for portability.

The final state-dict export uses the best checkpoint of the **last fold**,
matching the source script. It is not a fold ensemble or a refit on all data.
The bundled model is fold 5, epoch 299. `pvcracks/vae/training_results.json`
retains the supplied fold results; `model_manifest.json` records hashes and
the matching feature contract.

## XGBoost

`notebooks/train_xgboost.ipynb` replaces the older notebook in the runtime
package. Its reusable helper is `pvcracks_training.powerloss.training`:

```bash
python training/scripts/train_xgboost.py --table /data/features_and_targets.csv --output-dir outputs/xgboost
```

Generate features with `pvcracks.vae.features.extract_features(model, masks)`.
The table needs `latent_0` through `latent_319`, then crack/busbar/dark fractions
and component counts, plus measured `delta_pmp_percent`, `delta_voc_percent`,
and `Rsh`. Percentage targets use `100 * (post-pre) / pre`: negative means loss.
Do not apply PCA or scaling to the XGBoost inputs. Retraining the autoencoder
changes its representation; regenerate all embeddings and retrain XGBoost
together rather than mixing the new encoder with the supplied predictors.

The helper preserves the supplied 100-tree settings and 80/20 random row split
per target, drops missing target rows, and exports native model JSON plus
ordered feature metadata. Models remain fitted to the training partition.
The bundled example rows overlap the source training/evaluation data; their
predictions and reconstruction scores are usage examples, not independent
validation. For generalization to new modules, evaluate with a grouped split.

## MultiSolSegment

The existing Ray/HPO script remains in `scripts/train_channeled_unet_cv_hpo.py`;
its notebooks are now under `notebooks/multisolsegment/` and Slurm submission
is in `scripts/submit_hpo.slurm`. Configure data/output paths before running
these retained research experiments.
