"""Synthesize masks from a newly sampled empirical latent point and skip maps."""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from .features import prepare_masks
from .VAE_3CH_skipconn import threshold_probs


def synthesize_masks(model, masks, *, seed: int = 30, donor_count: int = 4):
    """Draw a new convex latent mixture and decode its synthesized feature maps.

    The current autoencoder requires spatial bottleneck and skip maps in
    addition to its pooled embedding. Draw distinct donors and Dirichlet
    weights, then apply the same weights to every feature level. Pooling the
    resulting bottleneck gives the sampled 320-value point exactly, up to
    floating-point precision. All skip maps are synthesized as well.

    This is empirical feature interpolation using the trained decoder. It is
    not sampling a learned Gaussian prior or generating a new raw EL photograph.
    The output is a new three-channel mask probability image; binary novelty
    relative to the supplied inputs can be checked by the caller.

    Returns a dict of CPU tensors: ``probabilities``, thresholded ``masks``,
    ``embedding``, ``donor_indices``, and ``weights``.
    """
    values = prepare_masks(masks)
    if not 2 <= donor_count <= len(values):
        raise ValueError("donor_count must be between 2 and the number of masks")
    generator = np.random.default_rng(seed)
    indices = generator.choice(len(values), size=donor_count, replace=False)
    weights = generator.dirichlet(np.full(donor_count, 2.0)).astype(np.float32)
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        blended = None
        for index, weight in zip(indices, weights):
            features = model.encode_feature_map(
                values[int(index) : int(index) + 1].to(device)
            )
            if blended is None:
                blended = {
                    name: value * float(weight) for name, value in features.items()
                }
            else:
                for name, value in features.items():
                    blended[name].add_(value, alpha=float(weight))
        embedding = F.adaptive_avg_pool2d(blended["b"], 1).flatten(1)
        probabilities = model.decode_from_features(blended)["probs"]
        return {
            "probabilities": probabilities.cpu(),
            "masks": threshold_probs(probabilities).cpu(),
            "embedding": embedding.cpu(),
            "donor_indices": torch.from_numpy(indices.copy()),
            "weights": torch.from_numpy(weights),
        }
