"""Configurable VAE training helpers kept out of the runtime package."""

from __future__ import annotations

import random
from collections.abc import Iterable

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from pvcracks.vae.VAE_model_1CH import VAE
from pvcracks.vae.pytorch_ssim import SSIM


def set_seeds(seed: int = 50) -> None:
    """Set Python, NumPy, and PyTorch seeds for a repeatable training run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_vae(
    latent_dim: int = 50,
    learning_rate: float = 1e-3,
    device: str | torch.device | None = None,
) -> tuple[VAE, Adam, torch.device]:
    """Create a one-channel VAE, Adam optimizer, and selected device."""
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = VAE(latent_dim).to(selected_device)
    return model, Adam(model.parameters(), lr=learning_rate), selected_device


def _vae_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    *,
    bce_weight: float,
    kld_weight: float,
    ssim_weight: float,
) -> torch.Tensor:
    """Compute the BCE, KL-divergence, and structural-similarity VAE loss."""
    bce = nn.functional.binary_cross_entropy(reconstruction, target, reduction="sum")
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    structural_loss = 1 - SSIM(window_size=50)(reconstruction, target)
    return bce_weight * bce + kld_weight * kl_divergence + ssim_weight * structural_loss


def train_vae(
    model: VAE,
    train_loader: Iterable[torch.Tensor],
    optimizer: Adam,
    *,
    epochs: int,
    device: str | torch.device,
    bce_weight: float = 0.01,
    kld_weight: float = 0.3,
    ssim_weight: float = 10000,
) -> list[float]:
    """Train a VAE and return mean epoch losses for user-owned data."""
    selected_device = torch.device(device)
    model.train()
    losses: list[float] = []
    for _ in range(epochs):
        total_loss = 0.0
        observations = 0
        for batch in train_loader:
            data = batch[0] if isinstance(batch, tuple) else batch
            data = data.to(selected_device)
            optimizer.zero_grad()
            reconstruction, mu, logvar = model(data)
            loss = _vae_loss(
                reconstruction, data, mu, logvar,
                bce_weight=bce_weight,
                kld_weight=kld_weight,
                ssim_weight=ssim_weight,
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            observations += len(data)
        if not observations:
            raise ValueError("train_loader did not yield any observations")
        losses.append(total_loss / observations)
    return losses
