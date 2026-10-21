"""
3-channel U-Net autoencoder PLUS for crack/busbar/dark mask reconstruction.

This version builds on the U-Net autoencoder that worked well.

Main improvements:
1. Separate output heads for crack, busbar, and dark.
2. BCEWithLogits losses for numerical stability.
3. Crack deep supervision at multiple decoder scales.
4. Moderate crack-specific losses:
   - Dice
   - Tversky
   - Multiscale Dice
   - Sobel edge loss
5. Balanced busbar/dark losses to preserve the channels that already work well.

Output directory:
    Set --output-dir when invoking train_3ch_autoencoder.py.
"""

import copy
import csv
import json
import os
import time as t
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from skimage import io
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from pvcracks.vae.VAE_3CH_skipconn import UNetAutoencoder3CHPlus
from pvcracks_training.vae.training import set_seeds

# =============================================================================
# Paths
# =============================================================================

# Paths are provided through main() / the command line, with no cwd changes.
bb_path = crack_path = dark_path = output_dir = None


# =============================================================================
# Hyperparameters
# =============================================================================

seed = 30
k = 5

batch_size = 4
num_epochs = 500

learning_rate = 8e-5
min_learning_rate = 1e-7
weight_decay = 1e-5
grad_clip_max_norm = 1.0

patience = 80
min_delta = 1e-5

# Increase to 24 if memory allows. Reduce to 16 if out-of-memory.
base_channels = 20

# Channel-level weights.
crack_global_weight = 1.35
busbar_global_weight = 1.0
dark_global_weight = 1.0

# Base per-channel losses.
crack_bce_weight = 1.0
crack_dice_weight = 2.0
crack_tversky_weight = 0.50
crack_multiscale_dice_weight = 0.50
crack_edge_weight = 0.30

busbar_bce_weight = 1.0
busbar_dice_weight = 1.0

dark_bce_weight = 1.0
dark_dice_weight = 1.0

# Positive weighting for sparse cracks.
# Keep this modest. Large values can make cracks too thick/noisy.
crack_pos_weight_value = 2.0

# Tversky parameters. alpha > beta favors recall.
crack_tversky_alpha = 0.60
crack_tversky_beta = 0.40

# Deep supervision for crack channel.
use_crack_deep_supervision = True
crack_deep_supervision_weight = 0.35

# Crack-aware sampler.
use_crack_weighted_sampler = True
sampler_crack_weight_scale = 5.0
sampler_max_weight = 8.0
sampler_num_samples_multiplier = 1.25

# Geometric augmentation.
use_random_batch_augmentation = True
hflip_prob = 0.5
vflip_prob = 0.5
rot180_prob = 0.5

# Thresholds for saved figures.
crack_display_threshold = 0.25
busbar_display_threshold = 0.50
dark_display_threshold = 0.50

num_example_cells_to_save = 6
log_interval_batches = 25


# =============================================================================
# Utility functions
# =============================================================================


def extract_prefix(filename):
    return int(filename.split("_")[0])


def finite_or_raise(name, tensor):
    if not torch.isfinite(tensor).all():
        safe_tensor = torch.nan_to_num(
            tensor.detach(),
            nan=0.0,
            posinf=1e20,
            neginf=-1e20,
        )
        msg = (
            f"{name} contains NaN or Inf. "
            f"safe min={safe_tensor.min().item():.6e}, "
            f"safe max={safe_tensor.max().item():.6e}"
        )
        raise FloatingPointError(msg)


def tensor_min_max_string(name, tensor):
    safe_tensor = torch.nan_to_num(
        tensor.detach(),
        nan=0.0,
        posinf=1e20,
        neginf=-1e20,
    )
    return (
        f"{name} min/max: "
        f"{safe_tensor.min().item():.6e} / {safe_tensor.max().item():.6e}"
    )


def get_batch_tensor(batch):
    if isinstance(batch, (tuple, list)):
        return batch[0]
    return batch


def parameters_are_finite(model):
    for name, param in model.named_parameters():
        if param is not None and not torch.isfinite(param).all():
            print(f"Non-finite parameter detected: {name}", flush=True)
            return False
    return True


def gradients_are_finite(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if not torch.isfinite(param.grad).all():
                print(f"Non-finite gradient detected: {name}", flush=True)
                return False
    return True


def random_batch_augment(data):
    if not use_random_batch_augmentation:
        return data

    augmented = data.clone()
    batch_size_local = augmented.size(0)
    device = augmented.device

    hmask = torch.rand(batch_size_local, device=device) < hflip_prob
    if hmask.any():
        augmented[hmask] = torch.flip(augmented[hmask], dims=[3])

    vmask = torch.rand(batch_size_local, device=device) < vflip_prob
    if vmask.any():
        augmented[vmask] = torch.flip(augmented[vmask], dims=[2])

    rmask = torch.rand(batch_size_local, device=device) < rot180_prob
    if rmask.any():
        augmented[rmask] = torch.flip(augmented[rmask], dims=[2, 3])

    return augmented


def make_crack_weighted_sampler(train_data):
    """
    Weighted sampler based on crack-pixel count.

    train_data shape:
        [N, 3, H, W]
    """
    crack_pixels = train_data[:, 0, :, :].reshape(train_data.shape[0], -1).sum(axis=1)
    log_counts = np.log1p(crack_pixels.astype(np.float64))

    if np.max(log_counts) > np.min(log_counts):
        normalized = (log_counts - np.min(log_counts)) / (
            np.max(log_counts) - np.min(log_counts)
        )
    else:
        normalized = np.zeros_like(log_counts)

    weights = 1.0 + sampler_crack_weight_scale * normalized
    weights = np.clip(weights, 1.0, sampler_max_weight)

    num_samples = int(np.ceil(len(train_data) * sampler_num_samples_multiplier))

    print("Crack weighted sampler statistics:", flush=True)
    print(f"  crack pixel count min: {float(np.min(crack_pixels)):.3f}", flush=True)
    print(f"  crack pixel count max: {float(np.max(crack_pixels)):.3f}", flush=True)
    print(f"  sampler weight min: {float(np.min(weights)):.3f}", flush=True)
    print(f"  sampler weight max: {float(np.max(weights)):.3f}", flush=True)
    print(f"  sampler num_samples: {num_samples}", flush=True)

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=num_samples,
        replacement=True,
    )

    return sampler


# =============================================================================
# Loss functions
# =============================================================================


def dice_loss_binary(prob_channel, target_channel, smooth=1.0):
    finite_or_raise("prob_channel in dice", prob_channel)
    finite_or_raise("target_channel in dice", target_channel)

    prob_flat = prob_channel.flatten(start_dim=1)
    target_flat = target_channel.flatten(start_dim=1)

    intersection = torch.sum(prob_flat * target_flat, dim=1)
    denominator = torch.sum(prob_flat, dim=1) + torch.sum(target_flat, dim=1)

    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return (1.0 - dice).mean()


def tversky_loss_binary(
    prob_channel, target_channel, alpha=0.60, beta=0.40, smooth=1.0
):
    prob_flat = prob_channel.flatten(start_dim=1)
    target_flat = target_channel.flatten(start_dim=1)

    true_pos = torch.sum(prob_flat * target_flat, dim=1)
    false_neg = torch.sum((1.0 - prob_flat) * target_flat, dim=1)
    false_pos = torch.sum(prob_flat * (1.0 - target_flat), dim=1)

    tversky = (true_pos + smooth) / (
        true_pos + alpha * false_neg + beta * false_pos + smooth
    )

    return (1.0 - tversky).mean()


def multiscale_dice_loss_binary(
    prob_channel, target_channel, kernel_sizes=(3, 5, 9), smooth=1.0
):
    losses = []

    for kernel_size in kernel_sizes:
        padding = kernel_size // 2

        prob_dilated = F.max_pool2d(
            prob_channel,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

        target_dilated = F.max_pool2d(
            target_channel,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )

        losses.append(dice_loss_binary(prob_dilated, target_dilated, smooth=smooth))

    return torch.stack(losses).mean()


def sobel_edge_map(x):
    dtype = x.dtype
    device = x.device

    sobel_x = torch.tensor(
        [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
        dtype=dtype,
        device=device,
    ).unsqueeze(0)

    sobel_y = torch.tensor(
        [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
        dtype=dtype,
        device=device,
    ).unsqueeze(0)

    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)

    return torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-8)


def sobel_edge_loss(prob_channel, target_channel):
    prob_edges = sobel_edge_map(prob_channel)
    target_edges = sobel_edge_map(target_channel)
    return F.l1_loss(prob_edges, target_edges)


def bce_logits_loss(logits, target, pos_weight_value=None):
    if pos_weight_value is None:
        return F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="mean",
        )

    pos_weight = torch.tensor(
        [pos_weight_value],
        dtype=logits.dtype,
        device=logits.device,
    )

    return F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=pos_weight,
        reduction="mean",
    )


def crack_deep_supervision_loss(crack_aux_logits, crack_target):
    if not use_crack_deep_supervision:
        return torch.zeros((), device=crack_target.device, dtype=crack_target.dtype)

    aux_weights = [0.50, 0.30, 0.20]
    total = torch.zeros((), device=crack_target.device, dtype=crack_target.dtype)

    for aux_logit, aux_weight in zip(crack_aux_logits, aux_weights):
        aux_logit_up = F.interpolate(
            aux_logit,
            size=crack_target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        aux_prob = torch.sigmoid(aux_logit_up)

        aux_bce = bce_logits_loss(
            aux_logit_up,
            crack_target,
            pos_weight_value=crack_pos_weight_value,
        )

        aux_dice = dice_loss_binary(aux_prob, crack_target)

        total = total + aux_weight * (aux_bce + aux_dice)

    return crack_deep_supervision_weight * total


def compute_total_loss(model_output, target_data):
    logits = model_output["logits"]
    probs = model_output["probs"]
    crack_aux_logits = model_output["crack_aux_logits"]

    finite_or_raise("logits", logits)
    finite_or_raise("probs", probs)
    finite_or_raise("target_data", target_data)

    target_data = target_data.float().clamp(0.0, 1.0)

    crack_target = target_data[:, 0:1, :, :]
    busbar_target = target_data[:, 1:2, :, :]
    dark_target = target_data[:, 2:3, :, :]

    crack_logits = logits[:, 0:1, :, :]
    busbar_logits = logits[:, 1:2, :, :]
    dark_logits = logits[:, 2:3, :, :]

    crack_prob = probs[:, 0:1, :, :]
    busbar_prob = probs[:, 1:2, :, :]
    dark_prob = probs[:, 2:3, :, :]

    # Crack losses.
    crack_bce = bce_logits_loss(
        crack_logits,
        crack_target,
        pos_weight_value=crack_pos_weight_value,
    )

    crack_dice = dice_loss_binary(crack_prob, crack_target)

    crack_tversky = tversky_loss_binary(
        crack_prob,
        crack_target,
        alpha=crack_tversky_alpha,
        beta=crack_tversky_beta,
    )

    crack_ms_dice = multiscale_dice_loss_binary(crack_prob, crack_target)
    crack_edge = sobel_edge_loss(crack_prob, crack_target)
    crack_deep = crack_deep_supervision_loss(crack_aux_logits, crack_target)

    crack_loss = crack_global_weight * (
        crack_bce_weight * crack_bce
        + crack_dice_weight * crack_dice
        + crack_tversky_weight * crack_tversky
        + crack_multiscale_dice_weight * crack_ms_dice
        + crack_edge_weight * crack_edge
        + crack_deep
    )

    # Busbar.
    busbar_bce = bce_logits_loss(busbar_logits, busbar_target)
    busbar_dice = dice_loss_binary(busbar_prob, busbar_target)

    busbar_loss = busbar_global_weight * (
        busbar_bce_weight * busbar_bce + busbar_dice_weight * busbar_dice
    )

    # Dark.
    dark_bce = bce_logits_loss(dark_logits, dark_target)
    dark_dice = dice_loss_binary(dark_prob, dark_target)

    dark_loss = dark_global_weight * (
        dark_bce_weight * dark_bce + dark_dice_weight * dark_dice
    )

    total_loss = crack_loss + busbar_loss + dark_loss

    finite_or_raise("total_loss", total_loss)

    diagnostics = {
        "total_loss": total_loss.detach(),
        "crack_loss": crack_loss.detach(),
        "busbar_loss": busbar_loss.detach(),
        "dark_loss": dark_loss.detach(),
        "crack_bce": crack_bce.detach(),
        "crack_dice": crack_dice.detach(),
        "crack_tversky": crack_tversky.detach(),
        "crack_multiscale_dice": crack_ms_dice.detach(),
        "crack_edge": crack_edge.detach(),
        "crack_deep": crack_deep.detach(),
        "busbar_bce": busbar_bce.detach(),
        "busbar_dice": busbar_dice.detach(),
        "dark_bce": dark_bce.detach(),
        "dark_dice": dark_dice.detach(),
    }

    return total_loss, diagnostics


# =============================================================================
# Saving and plotting helpers
# =============================================================================


def pad_loss_lists(loss_lists):
    max_len = max(len(x) for x in loss_lists)
    arr = np.full((len(loss_lists), max_len), np.nan, dtype=np.float64)

    for i, losses in enumerate(loss_lists):
        arr[i, : len(losses)] = losses

    return arr


def save_losses_csv(loss_arrays, output_dir):
    csv_path = output_dir / "training_validation_losses_total_and_channels.csv"

    mean_arrays = {
        "Average Train Total": np.nanmean(loss_arrays["train_total"], axis=0),
        "Average Val Total": np.nanmean(loss_arrays["val_total"], axis=0),
        "Average Train Crack": np.nanmean(loss_arrays["train_crack"], axis=0),
        "Average Val Crack": np.nanmean(loss_arrays["val_crack"], axis=0),
        "Average Train Busbar": np.nanmean(loss_arrays["train_busbar"], axis=0),
        "Average Val Busbar": np.nanmean(loss_arrays["val_busbar"], axis=0),
        "Average Train Dark": np.nanmean(loss_arrays["train_dark"], axis=0),
        "Average Val Dark": np.nanmean(loss_arrays["val_dark"], axis=0),
    }

    max_epochs = loss_arrays["train_total"].shape[1]

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        header = (
            ["Epoch"]
            + [f"Train Total Fold {i + 1}" for i in range(k)]
            + [f"Val Total Fold {i + 1}" for i in range(k)]
            + [f"Train Crack Fold {i + 1}" for i in range(k)]
            + [f"Val Crack Fold {i + 1}" for i in range(k)]
            + [f"Train Busbar Fold {i + 1}" for i in range(k)]
            + [f"Val Busbar Fold {i + 1}" for i in range(k)]
            + [f"Train Dark Fold {i + 1}" for i in range(k)]
            + [f"Val Dark Fold {i + 1}" for i in range(k)]
            + list(mean_arrays.keys())
        )

        writer.writerow(header)

        arrays = [
            loss_arrays["train_total"],
            loss_arrays["val_total"],
            loss_arrays["train_crack"],
            loss_arrays["val_crack"],
            loss_arrays["train_busbar"],
            loss_arrays["val_busbar"],
            loss_arrays["train_dark"],
            loss_arrays["val_dark"],
        ]

        for epoch in range(max_epochs):
            row = [epoch + 1]

            for arr in arrays:
                for fold_idx in range(k):
                    value = arr[fold_idx, epoch]
                    row.append(value if np.isfinite(value) else "N/A")

            for key in mean_arrays:
                value = mean_arrays[key][epoch]
                row.append(value if np.isfinite(value) else "N/A")

            writer.writerow(row)

    print(f"Saved loss CSV to {csv_path}", flush=True)


def save_loss_plots(loss_arrays, output_dir):
    train_total = np.nanmean(loss_arrays["train_total"], axis=0)
    val_total = np.nanmean(loss_arrays["val_total"], axis=0)

    train_crack = np.nanmean(loss_arrays["train_crack"], axis=0)
    val_crack = np.nanmean(loss_arrays["val_crack"], axis=0)

    train_busbar = np.nanmean(loss_arrays["train_busbar"], axis=0)
    val_busbar = np.nanmean(loss_arrays["val_busbar"], axis=0)

    train_dark = np.nanmean(loss_arrays["train_dark"], axis=0)
    val_dark = np.nanmean(loss_arrays["val_dark"], axis=0)

    epochs = np.arange(1, len(train_total) + 1)

    plt.figure(figsize=(9, 6))
    plt.plot(epochs, train_total, label="Train Total")
    plt.plot(epochs, val_total, label="Val Total")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Total Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    path_total = output_dir / "Training_Validation_Loss_Total.png"
    plt.savefig(path_total, dpi=200)
    plt.close()

    plt.figure(figsize=(11, 7))
    plt.plot(epochs, train_crack, label="Train Crack")
    plt.plot(epochs, val_crack, "--", label="Val Crack")
    plt.plot(epochs, train_busbar, label="Train Busbar")
    plt.plot(epochs, val_busbar, "--", label="Val Busbar")
    plt.plot(epochs, train_dark, label="Train Dark")
    plt.plot(epochs, val_dark, "--", label="Val Dark")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Channel Loss")
    plt.title("Training and Validation Loss by Channel")
    plt.legend()
    plt.tight_layout()
    path_channels = output_dir / "Training_Validation_Loss_Per_Channel.png"
    plt.savefig(path_channels, dpi=200)
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), squeeze=False)

    axes[0, 0].plot(epochs, train_total, label="Train Total")
    axes[0, 0].plot(epochs, val_total, label="Val Total")
    axes[0, 0].set_title("Total Loss")

    axes[0, 1].plot(epochs, train_crack, label="Train Crack")
    axes[0, 1].plot(epochs, val_crack, label="Val Crack")
    axes[0, 1].set_title("Crack Loss")

    axes[1, 0].plot(epochs, train_busbar, label="Train Busbar")
    axes[1, 0].plot(epochs, val_busbar, label="Val Busbar")
    axes[1, 0].set_title("Busbar Loss")

    axes[1, 1].plot(epochs, train_dark, label="Train Dark")
    axes[1, 1].plot(epochs, val_dark, label="Val Dark")
    axes[1, 1].set_title("Dark Loss")

    for row in axes:
        for ax in row:
            ax.set_yscale("log")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend(fontsize=8)

    plt.tight_layout()
    path_combined = output_dir / "Training_Validation_Loss_Total_and_Channels.png"
    plt.savefig(path_combined, dpi=200)
    plt.close()

    print(f"Saved total loss plot to {path_total}", flush=True)
    print(f"Saved per-channel loss plot to {path_channels}", flush=True)
    print(f"Saved combined loss plot to {path_combined}", flush=True)


def select_example_indices(dataset_length, num_examples):
    if dataset_length <= 0:
        raise ValueError("Cannot select examples from empty dataset.")

    n = min(num_examples, dataset_length)

    if n == 1:
        return [0]

    selected = np.linspace(0, dataset_length - 1, num=n, dtype=int)
    return sorted(list(set(selected.tolist())))


def sample_to_tensor(sample):
    sample = get_batch_tensor(sample)

    if not torch.is_tensor(sample):
        sample = torch.tensor(sample, dtype=torch.float32)

    if sample.ndim == 3:
        sample = sample.unsqueeze(0)

    return sample.float()


def get_reconstruction_for_sample(model, dataset, local_idx, device):
    model.eval()

    with torch.no_grad():
        sample = dataset[local_idx]
        sample = sample_to_tensor(sample)
        sample = sample.to(device).float().clamp(0.0, 1.0)

        output = model(sample)
        probs = output["probs"]

        finite_or_raise("example probs", probs)

        probs = probs.clamp(0.0, 1.0)

        input_np = sample[0].detach().cpu().numpy()
        recon_np = probs[0].detach().cpu().numpy()

    return input_np, recon_np


def save_selected_cell_examples(
    model,
    dataset,
    global_val_indices,
    output_dir,
    device,
    files_crack_sorted,
    files_bb_sorted,
    files_dark_sorted,
):
    selected_local_indices = select_example_indices(
        dataset_length=len(dataset),
        num_examples=num_example_cells_to_save,
    )

    selected_global_indices = [
        int(global_val_indices[local_idx]) for local_idx in selected_local_indices
    ]

    metadata_path = output_dir / "selected_cell_examples.csv"

    with open(metadata_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "example_number",
                "local_validation_index",
                "global_dataset_index",
                "crack_filename",
                "busbar_filename",
                "dark_filename",
            ]
        )

        for example_number, (local_idx, global_idx) in enumerate(
            zip(selected_local_indices, selected_global_indices),
            start=1,
        ):
            writer.writerow(
                [
                    example_number,
                    int(local_idx),
                    int(global_idx),
                    files_crack_sorted[int(global_idx)],
                    files_bb_sorted[int(global_idx)],
                    files_dark_sorted[int(global_idx)],
                ]
            )

    print(f"Saved selected cell metadata to {metadata_path}", flush=True)

    channel_info = [
        (0, "Crack", crack_display_threshold),
        (1, "Busbar", busbar_display_threshold),
        (2, "Dark", dark_display_threshold),
    ]

    n_examples = len(selected_local_indices)

    n_cols = 9

    fig, axes = plt.subplots(
        n_examples,
        n_cols,
        figsize=(24, max(3, 2.8 * n_examples)),
        squeeze=False,
    )

    for row_idx, (local_idx, global_idx) in enumerate(
        zip(selected_local_indices, selected_global_indices)
    ):
        input_np, recon_np = get_reconstruction_for_sample(
            model,
            dataset,
            local_idx,
            device,
        )

        col_idx = 0

        for channel_idx, channel_name, threshold in channel_info:
            input_channel = input_np[channel_idx]
            recon_channel = recon_np[channel_idx]
            threshold_channel = (recon_channel >= threshold).astype(np.float32)

            ax_input = axes[row_idx, col_idx]
            ax_prob = axes[row_idx, col_idx + 1]
            ax_thresh = axes[row_idx, col_idx + 2]

            ax_input.imshow(input_channel, cmap="gray", vmin=0.0, vmax=1.0)
            ax_prob.imshow(recon_channel, cmap="gray", vmin=0.0, vmax=1.0)
            ax_thresh.imshow(threshold_channel, cmap="gray", vmin=0.0, vmax=1.0)

            if row_idx == 0:
                ax_input.set_title(f"{channel_name}\nInput")
                ax_prob.set_title(f"{channel_name}\nProb")
                ax_thresh.set_title(f"{channel_name}\nThresh {threshold:.2f}")

            ax_input.set_ylabel(
                f"Cell {row_idx + 1}\nGlobal idx {int(global_idx)}",
                fontsize=9,
            )

            for ax in [ax_input, ax_prob, ax_thresh]:
                ax.set_xticks([])
                ax.set_yticks([])

            col_idx += 3

    plt.tight_layout()

    output_path = output_dir / "selected_cells_all_masks_input_prob_threshold.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(
        f"Saved selected-cell input/prob/threshold figure to {output_path}", flush=True
    )

    n_cols_simple = 6

    fig, axes = plt.subplots(
        n_examples,
        n_cols_simple,
        figsize=(18, max(3, 2.8 * n_examples)),
        squeeze=False,
    )

    for row_idx, (local_idx, global_idx) in enumerate(
        zip(selected_local_indices, selected_global_indices)
    ):
        input_np, recon_np = get_reconstruction_for_sample(
            model,
            dataset,
            local_idx,
            device,
        )

        col_idx = 0

        for channel_idx, channel_name, threshold in channel_info:
            input_channel = input_np[channel_idx]
            recon_channel = recon_np[channel_idx]

            ax_in = axes[row_idx, col_idx]
            ax_out = axes[row_idx, col_idx + 1]

            ax_in.imshow(input_channel, cmap="gray", vmin=0.0, vmax=1.0)
            ax_out.imshow(recon_channel, cmap="gray", vmin=0.0, vmax=1.0)

            if row_idx == 0:
                ax_in.set_title(f"{channel_name}\nInput")
                ax_out.set_title(f"{channel_name}\nProb Output")

            ax_in.set_ylabel(
                f"Cell {row_idx + 1}\nGlobal idx {int(global_idx)}",
                fontsize=9,
            )

            ax_in.set_xticks([])
            ax_in.set_yticks([])
            ax_out.set_xticks([])
            ax_out.set_yticks([])

            col_idx += 2

    plt.tight_layout()

    output_path_simple = output_dir / "selected_cells_all_masks_input_output.png"
    plt.savefig(output_path_simple, dpi=200)
    plt.close()

    print(
        f"Saved selected-cell input/output figure to {output_path_simple}", flush=True
    )


def save_best_parameters_log(output_dir, fold_summaries, elapsed_minutes):
    log_dict = {
        "script": "train_3ch_autoencoder.py",
        "output_dir": str(output_dir),
        "model": {
            "architecture": "UNetAutoencoder3CHPlus",
            "base_channels": base_channels,
            "note": (
                "Skip-connected autoencoder with separate heads "
                "and crack deep supervision."
            ),
        },
        "training": {
            "num_epochs_requested": num_epochs,
            "batch_size": batch_size,
            "k_folds": k,
            "seed": seed,
            "learning_rate": learning_rate,
            "min_learning_rate": min_learning_rate,
            "weight_decay": weight_decay,
            "patience": patience,
            "min_delta": min_delta,
            "grad_clip_max_norm": grad_clip_max_norm,
        },
        "loss": {
            "crack_global_weight": crack_global_weight,
            "busbar_global_weight": busbar_global_weight,
            "dark_global_weight": dark_global_weight,
            "crack_bce_weight": crack_bce_weight,
            "crack_dice_weight": crack_dice_weight,
            "crack_tversky_weight": crack_tversky_weight,
            "crack_multiscale_dice_weight": crack_multiscale_dice_weight,
            "crack_edge_weight": crack_edge_weight,
            "crack_deep_supervision_weight": crack_deep_supervision_weight,
            "crack_pos_weight_value": crack_pos_weight_value,
            "busbar_bce_weight": busbar_bce_weight,
            "busbar_dice_weight": busbar_dice_weight,
            "dark_bce_weight": dark_bce_weight,
            "dark_dice_weight": dark_dice_weight,
        },
        "sampler": {
            "use_crack_weighted_sampler": use_crack_weighted_sampler,
            "sampler_crack_weight_scale": sampler_crack_weight_scale,
            "sampler_max_weight": sampler_max_weight,
            "sampler_num_samples_multiplier": sampler_num_samples_multiplier,
        },
        "augmentation": {
            "use_random_batch_augmentation": use_random_batch_augmentation,
            "hflip_prob": hflip_prob,
            "vflip_prob": vflip_prob,
            "rot180_prob": rot180_prob,
        },
        "fold_summaries": fold_summaries,
        "elapsed_minutes": elapsed_minutes,
    }

    json_path = output_dir / "best_parameters_and_results.json"

    with open(json_path, "w") as f:
        json.dump(log_dict, f, indent=4)

    txt_path = output_dir / "best_parameters_and_results.txt"

    with open(txt_path, "w") as f:
        f.write("Best Parameters and Results\n")
        f.write("=" * 80 + "\n\n")
        f.write("Main idea\n")
        f.write("-" * 80 + "\n")
        f.write(
            "U-Net autoencoder with separate output heads "
            "and crack deep supervision.\n\n"
        )

        f.write("Key settings\n")
        f.write("-" * 80 + "\n")
        f.write(f"base_channels: {base_channels}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"learning_rate: {learning_rate}\n")
        f.write(f"weight_decay: {weight_decay}\n")
        f.write(f"use_crack_weighted_sampler: {use_crack_weighted_sampler}\n")
        f.write(f"crack_global_weight: {crack_global_weight}\n")
        f.write(f"crack_dice_weight: {crack_dice_weight}\n")
        f.write(f"crack_tversky_weight: {crack_tversky_weight}\n")
        f.write(f"crack_multiscale_dice_weight: {crack_multiscale_dice_weight}\n")
        f.write(f"crack_edge_weight: {crack_edge_weight}\n")
        f.write(f"crack_deep_supervision_weight: {crack_deep_supervision_weight}\n")
        f.write(f"crack_pos_weight_value: {crack_pos_weight_value}\n\n")

        f.write("Fold summaries\n")
        f.write("-" * 80 + "\n")
        for summary in fold_summaries:
            f.write(
                f"Fold {summary['fold']}: "
                f"best_epoch={summary['best_epoch']}, "
                f"best_val_total_loss={summary['best_val_total_loss']:.8e}, "
                f"best_val_crack_loss={summary['best_val_crack_loss']:.8e}, "
                f"best_val_busbar_loss={summary['best_val_busbar_loss']:.8e}, "
                f"best_val_dark_loss={summary['best_val_dark_loss']:.8e}, "
                f"epochs_ran={summary['epochs_ran']}, "
                f"final_lr={summary['final_lr']:.8e}\n"
            )

        f.write(f"\nElapsed minutes: {elapsed_minutes:.2f}\n")

    print(f"Saved parameter log to {json_path}", flush=True)
    print(f"Saved parameter log to {txt_path}", flush=True)


# =============================================================================
# Main script
# =============================================================================


def main(argv=None):
    """Run the supplied five-fold experiment with configurable data paths."""
    import argparse

    global bb_path, crack_path, dark_path, output_dir
    global seed, k, batch_size, num_epochs, base_channels, learning_rate
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crack-dir", type=Path, required=True)
    parser.add_argument("--busbar-dir", type=Path, required=True)
    parser.add_argument("--dark-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.base_channels) < 1 or args.folds < 2:
        parser.error(
            "epochs, batch size and base channels must be positive; folds >= 2"
        )
    if args.workers < 0 or args.learning_rate <= 0:
        parser.error("workers must be nonnegative and learning rate positive")
    crack_path, bb_path, dark_path = args.crack_dir, args.busbar_dir, args.dark_dir
    for directory in (crack_path, bb_path, dark_path):
        if not directory.is_dir():
            parser.error(f"Mask directory not found: {directory}")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    seed, k, batch_size = args.seed, args.folds, args.batch_size
    num_epochs, base_channels = args.epochs, args.base_channels
    learning_rate = args.learning_rate
    t0 = t.time()

    print("=" * 80, flush=True)
    print("Starting 3-channel U-Net PLUS autoencoder training", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print("=" * 80, flush=True)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    set_seeds(seed)

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}", flush=True)

    if device.type == "cuda":
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}", flush=True)

    print("Training settings:", flush=True)
    print(f"  base_channels = {base_channels}", flush=True)
    print(f"  batch_size = {batch_size}", flush=True)
    print(f"  learning_rate = {learning_rate}", flush=True)
    print(f"  use_crack_weighted_sampler = {use_crack_weighted_sampler}", flush=True)
    print(f"  use_crack_deep_supervision = {use_crack_deep_supervision}", flush=True)
    print(f"  output_dir = {output_dir}", flush=True)

    root_crack = Path(crack_path)
    root_bb = Path(bb_path)
    root_dark = Path(dark_path)

    files_crack = [
        x for x in os.listdir(root_crack) if x.lower().endswith((".tif", ".tiff"))
    ]
    files_bb = [x for x in os.listdir(root_bb) if x.lower().endswith((".tif", ".tiff"))]
    files_dark = [
        x for x in os.listdir(root_dark) if x.lower().endswith((".tif", ".tiff"))
    ]

    files_crack_sorted = sorted(files_crack, key=extract_prefix)
    files_bb_sorted = sorted(files_bb, key=extract_prefix)
    files_dark_sorted = sorted(files_dark, key=extract_prefix)

    print(f"Number of crack files: {len(files_crack_sorted)}", flush=True)
    print(f"Number of bb files:    {len(files_bb_sorted)}", flush=True)
    print(f"Number of dark files:  {len(files_dark_sorted)}", flush=True)

    if not (len(files_crack_sorted) == len(files_bb_sorted) == len(files_dark_sorted)):
        raise ValueError(
            "The crack, busbar, and dark directories must contain "
            "the same number of TIFF files."
        )

    prefixes = [
        [extract_prefix(name) for name in filenames]
        for filenames in (files_crack_sorted, files_bb_sorted, files_dark_sorted)
    ]
    if prefixes[0] != prefixes[1] or prefixes[0] != prefixes[2]:
        raise ValueError("Mask channel files must have matching numeric prefixes")
    if len(set(prefixes[0])) != len(prefixes[0]):
        raise ValueError("Mask filenames must have unique numeric prefixes")

    array_c = []
    array_b = []
    array_d = []

    print("Loading and thresholding images...", flush=True)

    for c, b, d in zip(files_crack_sorted, files_bb_sorted, files_dark_sorted):
        dat_c = io.imread(f"{crack_path}/{c}")
        datmean_c = dat_c.mean()
        dat_c = (dat_c > datmean_c).astype(np.float32)
        array_c.append(dat_c)

        dat_b = io.imread(f"{bb_path}/{b}")
        datmean_b = dat_b.mean()
        dat_b = (dat_b > datmean_b).astype(np.float32)
        array_b.append(dat_b)

        dat_d = io.imread(f"{dark_path}/{d}")
        datmean_d = dat_d.mean()
        dat_d = (dat_d > datmean_d).astype(np.float32)
        array_d.append(dat_d)

    combined = np.stack([array_c, array_b, array_d], axis=1).astype(np.float32)

    print(f"Combined array shape: {combined.shape}", flush=True)
    print(f"Combined min: {combined.min()}, max: {combined.max()}", flush=True)

    if combined.ndim != 4:
        raise ValueError(f"Expected combined shape [N, 3, H, W], got {combined.shape}")

    if combined.shape[1] != 3:
        raise ValueError(
            f"Expected 3 channels in dimension 1, got shape {combined.shape}"
        )

    if tuple(combined.shape[1:]) != (3, 400, 400):
        raise ValueError("Training masks must be 400 x 400 in all three channels")

    kfold = KFold(n_splits=k, shuffle=True, random_state=seed)

    fold_train_total_losses = []
    fold_val_total_losses = []

    fold_train_crack_losses = []
    fold_val_crack_losses = []

    fold_train_busbar_losses = []
    fold_val_busbar_losses = []

    fold_train_dark_losses = []
    fold_val_dark_losses = []

    fold_summaries = []

    last_val_dataset = None
    last_val_global_indices = None
    final_model = None

    for fold, (train_indices, val_indices) in enumerate(kfold.split(combined)):
        print("=" * 80, flush=True)
        print(f"Fold {fold + 1}/{k}", flush=True)
        print("=" * 80, flush=True)

        train_data = combined[train_indices]
        val_data = combined[val_indices]

        train_dataset = TensorDataset(torch.from_numpy(train_data))
        val_dataset = TensorDataset(torch.from_numpy(val_data))

        last_val_dataset = val_dataset
        last_val_global_indices = val_indices.copy()

        if use_crack_weighted_sampler:
            train_sampler = make_crack_weighted_sampler(train_data)
            train_shuffle = False
        else:
            train_sampler = None
            train_shuffle = True

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            num_workers=args.workers,
            pin_memory=(device.type == "cuda"),
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=(device.type == "cuda"),
        )

        model = UNetAutoencoder3CHPlus(
            in_channels=3,
            base=base_channels,
        )
        model.to(device)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=14,
            min_lr=min_learning_rate,
        )

        best_val_total_loss = float("inf")
        best_weights = None
        best_epoch = -1

        best_train_total_loss = float("inf")
        best_val_crack_loss = float("inf")
        best_val_busbar_loss = float("inf")
        best_val_dark_loss = float("inf")

        patience_counter = 0

        train_total_losses = []
        val_total_losses = []

        train_crack_losses = []
        val_crack_losses = []

        train_busbar_losses = []
        val_busbar_losses = []

        train_dark_losses = []
        val_dark_losses = []

        skipped_nonfinite_forward = 0
        skipped_nonfinite_grad = 0

        for epoch in range(num_epochs):
            model.train()

            epoch_train_total_sum = 0.0
            epoch_train_crack_sum = 0.0
            epoch_train_busbar_sum = 0.0
            epoch_train_dark_sum = 0.0

            num_successful_train_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                clean_data = get_batch_tensor(batch)
                clean_data = clean_data.to(device, non_blocking=True).float()
                clean_data = clean_data.clamp(0.0, 1.0)

                clean_target = random_batch_augment(clean_data)
                model_input = clean_target

                optimizer.zero_grad(set_to_none=True)

                try:
                    model_output = model(model_input)

                    loss, diagnostics = compute_total_loss(
                        model_output=model_output,
                        target_data=clean_target,
                    )

                except FloatingPointError as e:
                    skipped_nonfinite_forward += 1
                    print(
                        f"WARNING: Non-finite forward/loss at "
                        f"fold={fold + 1}, epoch={epoch + 1}, batch={batch_idx}. "
                        f"Skipping batch. Error: {e}",
                        flush=True,
                    )

                    optimizer.zero_grad(set_to_none=True)

                    if not parameters_are_finite(model):
                        raise RuntimeError(
                            "Model parameters became non-finite during forward pass."
                        )

                    continue

                loss.backward()

                if not gradients_are_finite(model):
                    skipped_nonfinite_grad += 1
                    print(
                        f"WARNING: Non-finite gradient at "
                        f"fold={fold + 1}, epoch={epoch + 1}, batch={batch_idx}. "
                        f"Skipping optimizer step.",
                        flush=True,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=grad_clip_max_norm,
                    error_if_nonfinite=False,
                )

                if not torch.isfinite(grad_norm):
                    skipped_nonfinite_grad += 1
                    print(
                        f"WARNING: grad_norm is non-finite at "
                        f"fold={fold + 1}, epoch={epoch + 1}, batch={batch_idx}. "
                        f"Skipping optimizer step.",
                        flush=True,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.step()

                if not parameters_are_finite(model):
                    raise RuntimeError(
                        f"Model parameters became non-finite after optimizer.step() at "
                        f"fold={fold + 1}, epoch={epoch + 1}, batch={batch_idx}."
                    )

                epoch_train_total_sum += diagnostics["total_loss"].item()
                epoch_train_crack_sum += diagnostics["crack_loss"].item()
                epoch_train_busbar_sum += diagnostics["busbar_loss"].item()
                epoch_train_dark_sum += diagnostics["dark_loss"].item()

                num_successful_train_batches += 1

                if batch_idx % log_interval_batches == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    probs = model_output["probs"]

                    print(
                        f"Fold {fold + 1}, Epoch {epoch + 1}, Batch {batch_idx}: "
                        f"total={diagnostics['total_loss'].item():.6e}, "
                        f"crack={diagnostics['crack_loss'].item():.6e}, "
                        f"busbar={diagnostics['busbar_loss'].item():.6e}, "
                        f"dark={diagnostics['dark_loss'].item():.6e}, "
                        f"lr={current_lr:.3e}, "
                        f"grad_norm={float(grad_norm):.6e}",
                        flush=True,
                    )
                    print(
                        "    " + tensor_min_max_string("input", model_input), flush=True
                    )
                    print(
                        "    " + tensor_min_max_string("target", clean_target),
                        flush=True,
                    )
                    print("    " + tensor_min_max_string("probs", probs), flush=True)
                    print(
                        f"    Crack components: "
                        f"BCE={diagnostics['crack_bce'].item():.6e}, "
                        f"Dice={diagnostics['crack_dice'].item():.6e}, "
                        f"Tversky={diagnostics['crack_tversky'].item():.6e}, "
                        f"MSDice={diagnostics['crack_multiscale_dice'].item():.6e}, "
                        f"Edge={diagnostics['crack_edge'].item():.6e}, "
                        f"Deep={diagnostics['crack_deep'].item():.6e}",
                        flush=True,
                    )

            if num_successful_train_batches == 0:
                raise RuntimeError(
                    f"No successful training batches in fold {fold + 1}, "
                    f"epoch {epoch + 1}."
                )

            epoch_train_total = epoch_train_total_sum / num_successful_train_batches
            epoch_train_crack = epoch_train_crack_sum / num_successful_train_batches
            epoch_train_busbar = epoch_train_busbar_sum / num_successful_train_batches
            epoch_train_dark = epoch_train_dark_sum / num_successful_train_batches

            train_total_losses.append(epoch_train_total)
            train_crack_losses.append(epoch_train_crack)
            train_busbar_losses.append(epoch_train_busbar)
            train_dark_losses.append(epoch_train_dark)

            # ---------------------------------------------------------------------
            # Validation
            # ---------------------------------------------------------------------
            model.eval()

            epoch_val_total_sum = 0.0
            epoch_val_crack_sum = 0.0
            epoch_val_busbar_sum = 0.0
            epoch_val_dark_sum = 0.0

            num_successful_val_batches = 0

            with torch.no_grad():
                for val_batch_idx, val_batch in enumerate(val_loader):
                    clean_val_data = get_batch_tensor(val_batch)
                    clean_val_data = clean_val_data.to(
                        device, non_blocking=True
                    ).float()
                    clean_val_data = clean_val_data.clamp(0.0, 1.0)

                    val_input = clean_val_data
                    val_target = clean_val_data

                    try:
                        val_output = model(val_input)

                        val_loss, val_diagnostics = compute_total_loss(
                            model_output=val_output,
                            target_data=val_target,
                        )

                    except FloatingPointError as e:
                        print(
                            f"WARNING: Non-finite validation batch at "
                            f"fold={fold + 1}, epoch={epoch + 1}, "
                            f"batch={val_batch_idx}. "
                            f"Skipping validation batch. Error: {e}",
                            flush=True,
                        )
                        continue

                    epoch_val_total_sum += val_diagnostics["total_loss"].item()
                    epoch_val_crack_sum += val_diagnostics["crack_loss"].item()
                    epoch_val_busbar_sum += val_diagnostics["busbar_loss"].item()
                    epoch_val_dark_sum += val_diagnostics["dark_loss"].item()

                    num_successful_val_batches += 1

            if num_successful_val_batches == 0:
                raise RuntimeError(
                    f"No successful validation batches in fold {fold + 1}, "
                    f"epoch {epoch + 1}."
                )

            epoch_val_total = epoch_val_total_sum / num_successful_val_batches
            epoch_val_crack = epoch_val_crack_sum / num_successful_val_batches
            epoch_val_busbar = epoch_val_busbar_sum / num_successful_val_batches
            epoch_val_dark = epoch_val_dark_sum / num_successful_val_batches

            val_total_losses.append(epoch_val_total)
            val_crack_losses.append(epoch_val_crack)
            val_busbar_losses.append(epoch_val_busbar)
            val_dark_losses.append(epoch_val_dark)

            scheduler.step(epoch_val_total)
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs}: "
                f"train_total={epoch_train_total:.6e}, "
                f"val_total={epoch_val_total:.6e}, "
                f"train_crack={epoch_train_crack:.6e}, "
                f"val_crack={epoch_val_crack:.6e}, "
                f"train_busbar={epoch_train_busbar:.6e}, "
                f"val_busbar={epoch_val_busbar:.6e}, "
                f"train_dark={epoch_train_dark:.6e}, "
                f"val_dark={epoch_val_dark:.6e}, "
                f"lr={current_lr:.3e}, "
                f"skipped_forward={skipped_nonfinite_forward}, "
                f"skipped_grad={skipped_nonfinite_grad}",
                flush=True,
            )

            if epoch_val_total < best_val_total_loss - min_delta:
                best_val_total_loss = epoch_val_total
                best_train_total_loss = epoch_train_total
                best_val_crack_loss = epoch_val_crack
                best_val_busbar_loss = epoch_val_busbar
                best_val_dark_loss = epoch_val_dark

                best_weights = copy.deepcopy(model.state_dict())
                best_epoch = epoch
                patience_counter = 0

                best_model_path = output_dir / f"best_model_fold_{fold + 1}.pth"

                torch.save(
                    {
                        "model_state_dict": best_weights,
                        "base_channels": base_channels,
                        "epoch": best_epoch + 1,
                        "best_val_total_loss": best_val_total_loss,
                        "best_train_total_loss": best_train_total_loss,
                        "best_val_crack_loss": best_val_crack_loss,
                        "best_val_busbar_loss": best_val_busbar_loss,
                        "best_val_dark_loss": best_val_dark_loss,
                        "learning_rate": learning_rate,
                    },
                    best_model_path,
                )

            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(
                    f"Early stopping in fold {fold + 1} at epoch {epoch + 1}.",
                    flush=True,
                )
                break

        if best_weights is not None:
            model.load_state_dict(best_weights)
            print(
                f"Restored fold {fold + 1} model to best epoch {best_epoch + 1} "
                f"with val_total_loss={best_val_total_loss:.6e}.",
                flush=True,
            )

        fold_train_total_losses.append(train_total_losses)
        fold_val_total_losses.append(val_total_losses)

        fold_train_crack_losses.append(train_crack_losses)
        fold_val_crack_losses.append(val_crack_losses)

        fold_train_busbar_losses.append(train_busbar_losses)
        fold_val_busbar_losses.append(val_busbar_losses)

        fold_train_dark_losses.append(train_dark_losses)
        fold_val_dark_losses.append(val_dark_losses)

        fold_summaries.append(
            {
                "fold": fold + 1,
                "best_epoch": best_epoch + 1,
                "best_val_total_loss": float(best_val_total_loss),
                "best_train_total_loss": float(best_train_total_loss),
                "best_val_crack_loss": float(best_val_crack_loss),
                "best_val_busbar_loss": float(best_val_busbar_loss),
                "best_val_dark_loss": float(best_val_dark_loss),
                "epochs_ran": len(train_total_losses),
                "final_lr": float(optimizer.param_groups[0]["lr"]),
                "skipped_nonfinite_forward": int(skipped_nonfinite_forward),
                "skipped_nonfinite_grad": int(skipped_nonfinite_grad),
            }
        )

        final_model = model

    # =============================================================================
    # Aggregate results
    # =============================================================================

    loss_arrays = {
        "train_total": pad_loss_lists(fold_train_total_losses),
        "val_total": pad_loss_lists(fold_val_total_losses),
        "train_crack": pad_loss_lists(fold_train_crack_losses),
        "val_crack": pad_loss_lists(fold_val_crack_losses),
        "train_busbar": pad_loss_lists(fold_train_busbar_losses),
        "val_busbar": pad_loss_lists(fold_val_busbar_losses),
        "train_dark": pad_loss_lists(fold_train_dark_losses),
        "val_dark": pad_loss_lists(fold_val_dark_losses),
    }

    print("=" * 80, flush=True)
    print(
        f"Average train total loss:   {np.nanmean(loss_arrays['train_total']):.6e}",
        flush=True,
    )
    print(
        f"Average val total loss:     {np.nanmean(loss_arrays['val_total']):.6e}",
        flush=True,
    )
    print(
        f"Average train crack loss:   {np.nanmean(loss_arrays['train_crack']):.6e}",
        flush=True,
    )
    print(
        f"Average val crack loss:     {np.nanmean(loss_arrays['val_crack']):.6e}",
        flush=True,
    )
    print(
        f"Average train busbar loss:  {np.nanmean(loss_arrays['train_busbar']):.6e}",
        flush=True,
    )
    print(
        f"Average val busbar loss:    {np.nanmean(loss_arrays['val_busbar']):.6e}",
        flush=True,
    )
    print(
        f"Average train dark loss:    {np.nanmean(loss_arrays['train_dark']):.6e}",
        flush=True,
    )
    print(
        f"Average val dark loss:      {np.nanmean(loss_arrays['val_dark']):.6e}",
        flush=True,
    )
    print("=" * 80, flush=True)

    save_losses_csv(loss_arrays, output_dir)
    save_loss_plots(loss_arrays, output_dir)

    # =============================================================================
    # Save final model and examples
    # =============================================================================

    if final_model is None:
        raise RuntimeError("No final model was produced.")

    final_model.to(device)
    final_model.eval()

    if last_val_dataset is not None and last_val_global_indices is not None:
        save_selected_cell_examples(
            model=final_model,
            dataset=last_val_dataset,
            global_val_indices=last_val_global_indices,
            output_dir=output_dir,
            device=device,
            files_crack_sorted=files_crack_sorted,
            files_bb_sorted=files_bb_sorted,
            files_dark_sorted=files_dark_sorted,
        )

    final_model_cpu = final_model.to("cpu")

    state_dict_path = (
        output_dir / f"model_3CH_{num_epochs}_UNET_AE_CRACK_PLUS_STATE_DICT.pth"
    )

    torch.save(
        {
            "model_state_dict": final_model_cpu.state_dict(),
            "base_channels": base_channels,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "min_learning_rate": min_learning_rate,
            "weight_decay": weight_decay,
            "crack_global_weight": crack_global_weight,
            "busbar_global_weight": busbar_global_weight,
            "dark_global_weight": dark_global_weight,
            "crack_bce_weight": crack_bce_weight,
            "crack_dice_weight": crack_dice_weight,
            "crack_tversky_weight": crack_tversky_weight,
            "crack_multiscale_dice_weight": crack_multiscale_dice_weight,
            "crack_edge_weight": crack_edge_weight,
            "crack_deep_supervision_weight": crack_deep_supervision_weight,
            "crack_pos_weight_value": crack_pos_weight_value,
            "busbar_bce_weight": busbar_bce_weight,
            "busbar_dice_weight": busbar_dice_weight,
            "dark_bce_weight": dark_bce_weight,
            "dark_dice_weight": dark_dice_weight,
            "use_crack_weighted_sampler": use_crack_weighted_sampler,
            "sampler_crack_weight_scale": sampler_crack_weight_scale,
            "sampler_max_weight": sampler_max_weight,
            "sampler_num_samples_multiplier": sampler_num_samples_multiplier,
            "use_random_batch_augmentation": use_random_batch_augmentation,
            "hflip_prob": hflip_prob,
            "vflip_prob": vflip_prob,
            "rot180_prob": rot180_prob,
            "crack_display_threshold": crack_display_threshold,
            "busbar_display_threshold": busbar_display_threshold,
            "dark_display_threshold": dark_display_threshold,
            "fold_summaries": fold_summaries,
        },
        state_dict_path,
    )

    print(f"Saved final state dict to {state_dict_path}", flush=True)

    full_model_path = output_dir / f"model_3CH_{num_epochs}_UNET_AE_CRACK_PLUS_FULL.pth"

    try:
        torch.save(final_model_cpu, full_model_path)
        print(f"Saved full model to {full_model_path}", flush=True)
    except Exception as e:
        print(
            f"WARNING: Could not save full model object. "
            f"State dict was saved successfully. Error: {e}",
            flush=True,
        )

    elapsed = t.time() - t0
    elapsed_minutes = elapsed / 60.0

    save_best_parameters_log(
        output_dir=output_dir,
        fold_summaries=fold_summaries,
        elapsed_minutes=elapsed_minutes,
    )

    print(
        f"Script completed successfully in {elapsed_minutes:.2f} minutes.", flush=True
    )
    print(f"All outputs saved to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
