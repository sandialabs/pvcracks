"""Prepare crack / busbar / dark masks and the matched XGBoost inputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from scipy import ndimage

CHANNELS = ("crack", "busbar", "dark")
IMAGE_SHAPE = (3, 400, 400)


def prepare_masks(masks) -> torch.Tensor:
    """Return float32 binary masks shaped ``[N, 3, 400, 400]``.

    Channels are crack, busbar, dark, in that order. Supply binary masks, not
    RGB EL images or reconstruction probabilities. Resizing / segmentation
    must happen before this step, as shown in the rapid-processing notebook.
    """
    values = torch.as_tensor(masks, dtype=torch.float32)
    if values.ndim == 3:
        values = values.unsqueeze(0)
    if values.ndim != 4 or tuple(values.shape[1:]) != IMAGE_SHAPE:
        raise ValueError(f"Expected [N, 3, 400, 400] masks; got {tuple(values.shape)}")
    if not len(values):
        raise ValueError("At least one mask is required")
    if not torch.all((values == 0) | (values == 1)):
        raise ValueError("Masks must contain binary values 0 or 1")
    return values


def morphology_features(masks) -> pd.DataFrame:
    """Measure original mask fractions and 8-connected component counts.

    The connectivity matches ``cv2.connectedComponents`` in the source
    analysis. Fractions and counts come from the input masks, before decoding.
    """
    values = prepare_masks(masks).detach().cpu().numpy()
    records = []
    for sample in values:
        record = {}
        for name, mask in zip(CHANNELS, sample):
            record[f"{name}_fraction"] = float(mask.mean())
        for name, mask in zip(CHANNELS, sample):
            record[f"{name}_components"] = int(
                ndimage.label(mask > 0.5, structure=np.ones((3, 3)))[1]
            )
        records.append(record)
    return pd.DataFrame(records)


def extract_features(model, masks, *, batch_size: int = 8) -> pd.DataFrame:
    """Build unscaled pooled embeddings plus the six morphology columns.

    Do not apply PCA or StandardScaler to this table before XGBoost prediction.
    ``model.encode`` is deterministic in eval mode and needs no VAE sampling.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    values = prepare_masks(masks)
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        embeddings = torch.cat(
            [model.encode(batch.to(device)).cpu() for batch in values.split(batch_size)]
        ).numpy()
    latent = pd.DataFrame(
        embeddings, columns=[f"latent_{i}" for i in range(embeddings.shape[1])]
    )
    return pd.concat([latent, morphology_features(values)], axis=1)
