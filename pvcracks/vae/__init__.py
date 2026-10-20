"""Pretrained three-channel crack / busbar / dark autoencoder.

The current model is a deterministic skip-connected autoencoder. The ``vae``
namespace is retained for existing imports; use ``model.encode`` for embeddings.
"""

from .VAE_3CH_skipconn import load_unet_ae_crack_plus

__all__ = ["load_unet_ae_crack_plus"]
