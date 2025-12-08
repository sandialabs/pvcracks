import os
import re
from typing import overload, Literal

import torch
from typing_extensions import OrderedDict

from . import img_functions
from .unet_model import construct_unet


@overload
def load_dataset(root, full_set: Literal[True]) -> img_functions.SolarDataset: ...


@overload
def load_dataset(
    root, full_set: Literal[False] = False
) -> tuple[img_functions.SolarDataset, img_functions.SolarDataset]: ...


def load_dataset(root, full_set=False):
    """Instantiate dataset objects with a common preprocessing pipeline.

    Args:
        root (str or Path): Root directory containing image and annotation folders.
        full_set (bool, optional): When `True`, return the combined dataset; otherwise
            return the train/validation split. Defaults to False.

    Returns:
        SolarDataset | tuple[SolarDataset, SolarDataset]: When `full_set=True`,
            returns a single SolarDataset with all data. When `full_set=False`,
            returns a tuple of (train_dataset, val_dataset).
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
    """Select an execution device and construct a DataParallel UNet model.

    Args:
        category_mapping (Mapping[str, int]): Mapping from class names to indices.
        existing_weight_path (str or Path, optional): Checkpoint file to load into the model.

    Returns:
        tuple[torch.device, torch.nn.Module]: Active device and initialized model module.
    """

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    unet = construct_unet(len(category_mapping))
    unet = torch.nn.DataParallel(unet)

    if existing_weight_path is not None:
        checkpoint = torch.load(existing_weight_path, map_location=device)

        # https://stackoverflow.com/questions/61909973/pytorch-load-incompatiblekeys
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = "module." + k
            new_state_dict[name] = v
        unet.load_state_dict(new_state_dict)

    model = unet.module.to(device)
    return device, model


def get_save_dir(base_dir, checkpoint_name) -> str:
    """Create and return the next sequential checkpoint directory path.

    Args:
        base_dir (str or Path): Base directory that contains the `checkpoints` subfolder.
        checkpoint_name (str): Prefix used when naming the new checkpoint directory.

    Returns:
        str: Absolute path to the newly created checkpoint directory.
    """
    checkpoint_dir = base_dir + "/checkpoints/"
    folders = [folder for folder in os.listdir(checkpoint_dir)]

    numbers = [int(re.search(r"(\d+)$", folder).group(1)) for folder in folders]
    next_number = max(numbers) + 1

    new_folder_name = f"{checkpoint_name}{next_number}"
    new_folder_path = os.path.join(checkpoint_dir, new_folder_name)

    os.makedirs(new_folder_path, exist_ok=True)

    return new_folder_path
