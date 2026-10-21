"""Inference, inspection, and visualization helpers for pretrained VAEs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from skimage import io
from torch.utils.data import DataLoader


def reconstruct_masks(model, masks, *, batch_size: int = 4) -> torch.Tensor:
    """Reconstruct binary 3-channel masks and return probabilities on the CPU."""
    from .features import prepare_masks

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    values = prepare_masks(masks)
    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        return torch.cat([
            model.reconstruct(batch.to(device)).cpu()
            for batch in values.split(batch_size)
        ])


def reconstruction_scores(masks, probabilities) -> pd.DataFrame:
    """Per-sample Dice and IoU using crack/busbar/dark thresholds .25/.50/.50."""
    from .features import CHANNELS, prepare_masks
    from .VAE_3CH_skipconn import threshold_probs

    target = prepare_masks(masks).cpu().bool()
    probs = torch.as_tensor(probabilities).detach().cpu()
    if probs.shape != target.shape or not torch.isfinite(probs).all():
        raise ValueError("Probabilities must be finite and match the input mask shape")
    prediction = threshold_probs(probs).bool()
    intersection = (prediction & target).sum(dim=(2, 3)).numpy()
    union = (prediction | target).sum(dim=(2, 3)).numpy()
    size = prediction.sum(dim=(2, 3)).numpy() + target.sum(dim=(2, 3)).numpy()
    scores = {}
    for channel, name in enumerate(CHANNELS):
        scores[f"{name}_dice"] = (2 * intersection[:, channel] + 1e-8) / (size[:, channel] + 1e-8)
        scores[f"{name}_iou"] = (intersection[:, channel] + 1e-8) / (union[:, channel] + 1e-8)
    return pd.DataFrame(scores)


def preprocess(image_path: str | Path) -> np.ndarray:
    """Read an EL image and return its first two channels scaled to [0, 1]."""
    image = io.imread(image_path)
    return (image[:, :, :2] / 255).astype("float32")


def encode_image(model: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
    """Return a sampled latent representation for one prepared image."""
    model.eval()
    with torch.no_grad():
        if hasattr(model, "encode"):
            return model.encode(image.unsqueeze(0))
        mu, logvar = model.encoder(image.unsqueeze(0))
        return model.reparameterize(mu, logvar)


def decode_latent_vector(model: torch.nn.Module, latent_vector: torch.Tensor) -> torch.Tensor:
    """Decode a latent vector with a pretrained VAE."""
    if not hasattr(model, "decoder"):
        raise ValueError("The skip-connected autoencoder needs encoder features to reconstruct masks")
    model.eval()
    with torch.no_grad():
        return model.decoder(latent_vector)


def load_from_testloader(test, num_images: int = 100, *, device=None) -> torch.Tensor:
    """Return a random batch from a test dataset on the requested device."""
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    images = next(iter(DataLoader(test, batch_size=num_images, shuffle=True, num_workers=4)))
    return images.to(selected_device)


def VAE_output_for_images(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Reconstruct images with a pretrained VAE."""
    model.eval()
    with torch.no_grad():
        result = model(images)
        outputs = result["probs"] if isinstance(result, dict) else result[0]
    return outputs


def show_input_output_images(inputs, outputs, num_images: int, path: str | Path | None = None):
    """Plot input/reconstruction pairs and annotate their structural similarity."""
    from skimage.metrics import structural_similarity

    inputs = inputs.detach().cpu()
    outputs = outputs.detach().cpu()
    figure, axes = plt.subplots(num_images, 2, figsize=(8, 4 * num_images), squeeze=False)
    for index in range(num_images):
        source = inputs[index].squeeze().numpy()
        reconstruction = outputs[index].squeeze().numpy()
        score = structural_similarity(source, reconstruction, data_range=1.0)
        axes[index, 0].imshow(source, cmap="gray")
        axes[index, 0].set_axis_off()
        axes[index, 1].imshow(reconstruction, cmap="gray")
        axes[index, 1].set_title(f"SSIM: {score:.3f}")
        axes[index, 1].set_axis_off()
    figure.tight_layout()
    if path:
        figure.savefig(Path(path) / "example_input_output.png", dpi=150)
    return figure


def ssim_input_output(inputs, outputs, num_images: int, path: str | Path | None = None) -> pd.DataFrame:
    """Calculate structural-similarity scores for VAE reconstructions."""
    from skimage.metrics import structural_similarity

    inputs = inputs.detach().cpu()
    outputs = outputs.detach().cpu()
    scores = [
        structural_similarity(
            inputs[index].squeeze().numpy(),
            outputs[index].squeeze().numpy(),
            data_range=1.0,
        )
        for index in range(num_images)
    ]
    result = pd.DataFrame({"num": range(num_images), "ssim_score": scores})
    if path:
        destination = Path(path)
        result.to_csv(destination / "ssim_comp.csv", index=False)
        axis = result["ssim_score"].plot(kind="box", title="VAE structural similarity")
        axis.axhline(y=0.95, color="r", linestyle="--")
        axis.figure.savefig(destination / "SSIM_compare.png", dpi=150)
    return result


def generate_random_images(
    model: torch.nn.Module,
    num_images: int,
    latent_dim: int,
    *,
    device=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random latent vectors and decode them with a pretrained VAE."""
    selected_device = torch.device(device or next(model.parameters()).device)
    if not hasattr(model, "decoder"):
        raise ValueError("The skip-connected autoencoder reconstructs input masks; it cannot decode random embeddings")
    model.eval()
    latent_vectors = torch.randn(num_images, latent_dim, device=selected_device)
    with torch.no_grad():
        images = model.decoder(latent_vectors)
    return images, latent_vectors


def show_generated_images(generated_images, num_images: int, path: str | Path | None = None):
    """Plot generated single-channel VAE images."""
    num_cols = 5
    num_rows = (num_images + num_cols - 1) // num_cols
    figure, axes = plt.subplots(num_rows, num_cols, figsize=(2 * num_cols, 2 * num_rows))
    for axis, image in zip(np.ravel(axes), generated_images.detach().cpu()):
        axis.imshow(image.squeeze().numpy(), cmap="gray")
        axis.set_axis_off()
    for axis in np.ravel(axes)[num_images:]:
        axis.set_axis_off()
    figure.tight_layout()
    if path:
        figure.savefig(Path(path) / "GenImages.png", dpi=150)
    return figure
