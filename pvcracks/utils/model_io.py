"""Reusable MultiSolSegment data and pretrained-model loading helpers.

The functions in this module are deliberately limited to preparing a labelled
dataset and loading a pretrained model. They are used by the documentation
notebooks and remain in the lightweight :mod:`pvcracks` distribution. Model
retraining and experiment-management utilities are provided by the separate
``pvcracks-training`` distribution.
"""

from collections import OrderedDict
from typing import Literal, overload

import torch

from . import img_functions
from .unet_model import construct_unet


@overload
def load_dataset(root, full_set: Literal[True]) -> img_functions.SolarDataset: ...


@overload
def load_dataset(
    root, full_set: Literal[False] = False
) -> tuple[img_functions.SolarDataset, img_functions.SolarDataset]: ...


def load_dataset(root, full_set=False):
    """Instantiate labelled MultiSolSegment datasets with common transforms.

    Args:
        root: Directory containing the ``img`` and ``ann`` data directories.
        full_set: Return the ``all`` partition when true; otherwise return the
            ``train`` and ``val`` partitions.
    """
    transformers = img_functions.Compose(
        [
            img_functions.ChanneledFixResize(256),
            img_functions.ToTensor(),
            img_functions.Normalize(),
        ]
    )

    if full_set:
        return img_functions.SolarDataset(
            root, image_folder="img/all", mask_folder="ann/all", transforms=transformers
        )

    train_dataset = img_functions.SolarDataset(
        root, image_folder="img/train", mask_folder="ann/train", transforms=transformers
    )
    val_dataset = img_functions.SolarDataset(
        root, image_folder="img/val", mask_folder="ann/val", transforms=transformers
    )
    return train_dataset, val_dataset


def load_device_and_model(
    category_mapping, existing_weight_path=None
) -> tuple[torch.device, torch.nn.Module]:
    """Select a device and construct a MultiSolSegment model for inference.

    Args:
        category_mapping: Mapping from segmentation categories to indices.
        existing_weight_path: Optional model checkpoint to load.

    Returns:
        The selected device and a model moved to that device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = torch.nn.DataParallel(construct_unet(len(category_mapping)))
    if existing_weight_path is not None:
        checkpoint = torch.load(existing_weight_path, map_location=device)
        state_dict = OrderedDict((f"module.{key}", value) for key, value in checkpoint.items())
        model.load_state_dict(state_dict)

    return device, model.module.to(device)
