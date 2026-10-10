"""Utilities shared by optional MultiSolSegment retraining workflows."""

from __future__ import annotations

import re
from pathlib import Path

from pvcracks.utils.model_io import load_dataset


def create_checkpoint_directory(base_dir: str | Path, checkpoint_name: str) -> Path:
    """Create the next numbered checkpoint directory below ``base_dir``.

    Non-checkpoint files such as ``.DS_Store`` are ignored, making this helper
    suitable for local workstations and shared HPC filesystems.
    """
    checkpoint_root = Path(base_dir) / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(checkpoint_name)}(\d+)$")
    existing = [
        int(match.group(1))
        for path in checkpoint_root.iterdir()
        if (match := pattern.match(path.name))
    ]
    checkpoint_dir = checkpoint_root / f"{checkpoint_name}{max(existing, default=0) + 1}"
    checkpoint_dir.mkdir()
    return checkpoint_dir


def load_training_datasets(root: str | Path, *, full_set: bool = False):
    """Load labelled MultiSolSegment data for a training workflow."""
    return load_dataset(root, full_set=full_set)
