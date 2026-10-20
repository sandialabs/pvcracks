"""
VAE_3CH_skipconn.py

Reusable model definition for the successful 3-channel skip-connected U-Net
autoencoder used for crack / busbar / dark mask reconstruction.

Important:
    This is not a classical VAE. It is a U-Net-style autoencoder with skip
    connections and separate output heads.

    For latent-space analysis, use model.encode(x), which returns a compact
    bottleneck embedding obtained by global-average-pooling the bottleneck
    feature map.

Expected input::

    x shape: [B, 3, 400, 400]

Output from forward::

    {
        "logits": [B, 3, 400, 400],
        "probs": [B, 3, 400, 400],
        "crack_aux_logits": list of auxiliary crack logits
    }

Channel order::

    0: crack
    1: busbar
    2: dark
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Two Conv-BatchNorm-LeakyReLU blocks.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class OutputHead(nn.Module):
    """
    Small output head used for each channel.
    """

    def __init__(self, in_channels, hidden_channels, out_channels=1):
        super().__init__()

        self.head = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.head(x)


class UNetAutoencoder3CHPlus(nn.Module):
    """
    U-Net-style autoencoder for 3-channel 400 x 400 binary masks.

    Down path:
        400 -> 200 -> 100 -> 50 -> 25

    Up path:
        25 -> 50 -> 100 -> 200 -> 400

    Final output:
        separate crack, busbar, and dark heads.

    Latent representation:
        The encode() method returns global-average-pooled bottleneck features.
        For base=20, this gives a vector of length 320 because bottleneck has
        16 * base channels.
    """

    def __init__(self, in_channels=3, base=20):
        super().__init__()

        self.in_channels = in_channels
        self.base = base
        self.latent_dim = base * 16

        # Encoder
        self.enc1 = DoubleConv(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(base * 4, base * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base * 8, base * 16)

        # Decoder
        self.up4 = nn.ConvTranspose2d(
            base * 16,
            base * 8,
            kernel_size=2,
            stride=2,
        )
        self.dec4 = DoubleConv(base * 16, base * 8)

        self.up3 = nn.ConvTranspose2d(
            base * 8,
            base * 4,
            kernel_size=2,
            stride=2,
        )
        self.dec3 = DoubleConv(base * 8, base * 4)

        self.up2 = nn.ConvTranspose2d(
            base * 4,
            base * 2,
            kernel_size=2,
            stride=2,
        )
        self.dec2 = DoubleConv(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(
            base * 2,
            base,
            kernel_size=2,
            stride=2,
        )
        self.dec1 = DoubleConv(base * 2, base)

        # Separate final output heads
        self.crack_head = OutputHead(base, base, out_channels=1)
        self.busbar_head = OutputHead(base, base, out_channels=1)
        self.dark_head = OutputHead(base, base, out_channels=1)

        # Crack auxiliary heads for deep supervision
        self.crack_aux_200 = nn.Conv2d(base * 2, 1, kernel_size=1)
        self.crack_aux_100 = nn.Conv2d(base * 4, 1, kernel_size=1)
        self.crack_aux_50 = nn.Conv2d(base * 8, 1, kernel_size=1)

    def encode_feature_map(self, x):
        """
        Run the encoder and return the bottleneck feature map plus skip features.

        Returns
        -------
        dict
            Encoder maps ``e1`` (400 x 400), ``e2`` (200 x 200), ``e3``
            (100 x 100), ``e4`` (50 x 50), and ``b`` (25 x 25).
        """
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        b = self.bottleneck(p4)

        return {
            "e1": e1,
            "e2": e2,
            "e3": e3,
            "e4": e4,
            "b": b,
        }

    def encode(self, x, flatten=True):
        """
        Return a compact latent-like embedding from the bottleneck.

        This is intended for PCA, UMAP, t-SNE, clustering, and similarity
        analysis.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [B, 3, 400, 400].
        flatten : bool
            If True, return global-average-pooled bottleneck vector:
                [B, 16 * base]
            If False, return full bottleneck feature map:
                [B, 16 * base, 25, 25]

        Returns
        -------
        torch.Tensor
            Latent-like representation.
        """
        features = self.encode_feature_map(x)
        b = features["b"]

        if not flatten:
            return b

        z = F.adaptive_avg_pool2d(b, output_size=(1, 1))
        z = z.flatten(start_dim=1)

        return z

    def decode_from_features(self, features):
        """
        Decode from encoder feature maps using skip connections.

        This is primarily used internally by forward().
        """
        e1 = features["e1"]
        e2 = features["e2"]
        e3 = features["e3"]
        e4 = features["e4"]
        b = features["b"]

        u4 = self.up4(b)
        d4 = self.dec4(torch.cat([u4, e4], dim=1))

        u3 = self.up3(d4)
        d3 = self.dec3(torch.cat([u3, e3], dim=1))

        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        crack_logits = self.crack_head(d1)
        busbar_logits = self.busbar_head(d1)
        dark_logits = self.dark_head(d1)

        logits = torch.cat(
            [crack_logits, busbar_logits, dark_logits],
            dim=1,
        )

        crack_aux_logits = [
            self.crack_aux_200(d2),
            self.crack_aux_100(d3),
            self.crack_aux_50(d4),
        ]

        return {
            "logits": logits,
            "probs": torch.sigmoid(logits),
            "crack_aux_logits": crack_aux_logits,
            "decoder_features": {
                "d1": d1,
                "d2": d2,
                "d3": d3,
                "d4": d4,
            },
        }

    def forward(self, x):
        features = self.encode_feature_map(x)
        output = self.decode_from_features(features)

        # Include bottleneck info for optional diagnostics.
        output["bottleneck_feature_map"] = features["b"]
        output["embedding"] = F.adaptive_avg_pool2d(
            features["b"],
            output_size=(1, 1),
        ).flatten(start_dim=1)

        return output

    def reconstruct(self, x):
        """
        Convenience method returning only probabilities.
        """
        return self.forward(x)["probs"]


def build_unet_ae_crack_plus(base_channels=20, in_channels=3):
    """
    Build an uninitialized UNetAutoencoder3CHPlus.
    """
    return UNetAutoencoder3CHPlus(
        in_channels=in_channels,
        base=base_channels,
    )


def load_unet_ae_crack_plus(
    checkpoint_path=None,
    device=None,
    base_channels=None,
    strict=True,
):
    """
    Load UNetAutoencoder3CHPlus from a state-dict checkpoint.

    This expects the checkpoint format saved by the training script::

        {
            "model_state_dict": ...,
            "base_channels": ...,
            ...
        }

    Parameters
    ----------
    checkpoint_path : str or pathlib.Path or None
        State-dict checkpoint. None uses the matching bundled model.
    device : torch.device or str or None
        Device to load onto. If None, auto-selects CUDA if available.
    base_channels : int or None
        If None, uses checkpoint["base_channels"] when available.
    strict : bool
        Passed to model.load_state_dict.

    Returns
    -------
    model : UNetAutoencoder3CHPlus
    checkpoint : dict
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_path is None:
        checkpoint_path = Path(__file__).with_name(
            "model_3CH_500_UNET_AE_CRACK_PLUS_STATE_DICT.pth"
        )
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    if base_channels is None:
        base_channels = checkpoint.get("base_channels", 20)

    model = UNetAutoencoder3CHPlus(
        in_channels=3,
        base=base_channels,
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        # Allows loading a raw state_dict if needed.
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    model.eval()

    return model, checkpoint


def tensor_to_numpy_image(tensor):
    """
    Convenience helper for plotting a single [3, H, W] tensor.
    """
    if tensor.ndim == 4:
        tensor = tensor[0]

    return tensor.detach().cpu().numpy()


def threshold_probs(probs, thresholds=(0.25, 0.50, 0.50)):
    """
    Threshold probability maps channel-wise.

    Parameters
    ----------
    probs : torch.Tensor
        Shape [B, 3, H, W] or [3, H, W].
    thresholds : tuple
        Thresholds for crack, busbar, dark.

    Returns
    -------
    torch.Tensor
        Binary tensor with same shape as probs.
    """
    squeeze_back = False

    if probs.ndim == 3:
        probs = probs.unsqueeze(0)
        squeeze_back = True

    out = torch.zeros_like(probs)

    for channel_idx, threshold in enumerate(thresholds):
        out[:, channel_idx : channel_idx + 1] = (
            probs[:, channel_idx : channel_idx + 1] >= threshold
        ).float()

    if squeeze_back:
        out = out[0]

    return out
