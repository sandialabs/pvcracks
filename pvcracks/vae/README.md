# Current 3-channel model

`VAE_3CH_skipconn.UNetAutoencoder3CHPlus` is the FinalVAE3CH skip-connected
autoencoder supplied by Norman Jost. It is a deterministic autoencoder, with
no KL loss, latent sampling, or standalone decoder for random embeddings.
The `vae` package name and older class modules remain for historical imports.

```python
from pvcracks.vae import load_unet_ae_crack_plus
from pvcracks.vae.features import extract_features
from pvcracks.powerloss.powerloss_functions import load_all_xgb_models

model, checkpoint = load_unet_ae_crack_plus(device="cpu")
# masks: [N, 3, 400, 400], binary crack / busbar / dark channels
features = extract_features(model, masks)
models = load_all_xgb_models()
predictions = {target: predictor.predict(features) for target, predictor in models.items()}
```

The bundled state dictionary is the source script's final export: best
validation weights from fold 5, epoch 299, with base width 20. It is not the
best fold across all five, an ensemble, or an all-data refit. See
`training_results.json` for the supplied training settings and fold results,
and `model_manifest.json` for source identity and SHA-256 hashes.

Each XGBoost model uses 326 features: `latent_0` through `latent_319` from
global average pooling, three input-mask fractions, then three 8-connected
component counts. Channels are always crack, busbar, dark. Features are
unscaled and unprojected. Model JSON and sibling metadata are packaged under
`pvcracks.powerloss`. DataFrame columns are selected and ordered by name.
The models estimate signed ΔPmp (%), signed ΔVoc (%), and Rsh; negative
percentage changes indicate losses. They preserve the source analysis's target
definitions, training split, and predictions. Rsh uses the source I-V units.

The updated examples include 26 source masks and reference export values.
Predictions on these rows demonstrate usage, not independent validation.
Pooling differences across PyTorch backends can move embeddings near a tree
split; frozen exported features reproduce the source XGBoost predictions.
Retraining instructions are in `training/README.md`.

The image-generation notebook samples a new empirical latent point with
`pvcracks.vae.synthesis.synthesize_masks`. It uses random convex combinations
of multiple encoded examples and applies the same combination to the spatial
bottleneck and every skip map before decoding a new mask image. This feature
interpolation does not introduce a learned probabilistic prior. The notebook
checks that the sampled point and binary output differ from the input examples.
