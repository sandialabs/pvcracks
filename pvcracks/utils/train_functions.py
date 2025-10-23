import os
from typing import Union

import torch
from typing_extensions import OrderedDict

from .unet_model import construct_unet
from . import img_functions


def load_dataset(
    root, full_set=False
) -> Union[
    img_functions.SolarDataset,
    tuple[img_functions.SolarDataset, img_functions.SolarDataset],
]:
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
    
    # NOTE: change this based on available hardware
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.mps.is_available() else "cpu")

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
    checkpoint_dir = base_dir + "/checkpoints/"
    folders = [folder for folder in os.listdir(checkpoint_dir)]

    max_number = 0
    for folder in folders:
        number = int(folder[-1])
        if number > max_number:
            max_number = number

    new_folder_name = f"{checkpoint_name}{max_number + 1}"
    new_folder_path = os.path.join(checkpoint_dir, new_folder_name)

    os.makedirs(new_folder_path, exist_ok=True)

    return new_folder_path
