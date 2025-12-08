import os
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
import torchvision.transforms.functional as F
from imutils.paths import list_files
from PIL import Image
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset


class SolarDataset(VisionDataset):
    """A dataset for our solar panel images and masks."""

    def __init__(
        self, root, image_folder, mask_folder, transforms, mode="train", random_seed=42
    ):
        """Set up the dataset by collecting and shuffling image and mask paths.

        Args:
            root (str or Path): Root directory that contains the data folders.
            image_folder (str or Path): Subdirectory with image files.
            mask_folder (str or Path): Subdirectory with mask files stored as numpy arrays.
            transforms (callable): Callable applied to `(image, mask)` during `__getitem__`.
            mode (str, optional): Dataset split indicator; kept for backward compatibility.
            random_seed (int, optional): Seed used when shuffling to keep pairs aligned.
        """
        super().__init__(root, transforms)

        self.image_path = Path(self.root) / image_folder
        self.mask_path = Path(self.root) / mask_folder

        if not os.path.exists(self.image_path):
            raise OSError(f"{self.image_path} not found.")

        if not os.path.exists(self.mask_path):
            raise OSError(f"{self.mask_path} not found.")

        self.image_list = sorted(
            [
                c
                for c in list(list_files(self.image_path))
                if ".ipynb_checkpoints" not in c
            ]
        )
        self.mask_list = sorted(
            [
                c
                for c in list(list_files(self.mask_path))
                if ".ipynb_checkpoints" not in c
            ]
        )
        self.image_list = np.array(self.image_list)
        self.mask_list = np.array(self.mask_list)

        np.random.seed(random_seed)
        index = np.arange(len(self.image_list))
        np.random.shuffle(index)
        self.image_list = self.image_list[index]
        self.mask_list = self.mask_list[index]

    def __len__(self):
        """Return the total number of samples available."""
        return len(self.image_list)

    def get_img_path(self, index):
        """Return the full path to the image file at `index`.

        Args:
            index (int): Dataset index referencing the desired image.

        Returns:
            str: Absolute path to the image file.
        """
        return self.image_list[index]

    def __get_mask_path__(self, index):
        """Return the full path to the mask file at `index`.

        Args:
            index (int): Dataset index referencing the desired mask.

        Returns:
            str: Absolute path to the mask file.
        """
        return self.mask_list[index]

    def __getname__(self, index):
        """Return the name of the image and mask at the given index.

        Args:
            index (int): The index of the image and mask.

        Returns:
            str: The name of the image and mask.

        Raises:
            IndexError: If the filenames of the image and mask do not match.
        """
        image_name = os.path.splitext(os.path.split(self.image_list[index])[-1])[0]
        mask_name = os.path.splitext(os.path.split(self.mask_list[index])[-1])[0]

        if image_name == mask_name:
            return image_name
        else:
            return IndexError("Image and mask names do not match.")

    def __getraw__(self, index) -> tuple[Image.Image, np.ndarray]:
        """Load the raw PIL image and numpy mask for the given index.

        Args:
            index (int): Dataset index referencing the desired sample.

        Returns:
            tuple: `(PIL.Image.Image, numpy.ndarray)` for the image and mask.

        Raises:
            ValueError: If the filenames of the image and mask do not match.
        """
        if not self.__getname__(index):
            raise ValueError(
                "{}: Image doesn't match with mask".format(
                    os.path.split(self.image_list[index])[-1]
                )
            )
        image = Image.open(self.image_list[index])
        mask = np.load(self.mask_list[index], allow_pickle=True)

        return image, mask

    def __getitem__(self, index):
        """Load and transform the sample identified by `index`.

        Args:
            index (int): Dataset index referencing the desired sample.

        Returns:
            tuple: Transformed `(image, mask)` pair ready for the model.
        """
        image, mask = self.__getraw__(index)
        image, mask = self.transforms(image, mask)

        return image, mask


class Compose:
    def __init__(self, transforms):
        """Store a sequence of paired image/mask transforms.

        Args:
            transforms (Iterable[callable]): Transform callables accepting `(image, mask)`.
        """
        self.transforms = transforms

    def __call__(self, image, target):
        """Sequentially apply all stored transforms to the `(image, target)` pair.

        Args:
            image: Input image passed to each transform.
            target: Segmentation mask passed to each transform.

        Returns:
            tuple: The transformed `(image, target)` pair.
        """
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


# MODIFIED FOR NUMPY ARRAY INPUT
class FixResize:
    """Resize PIL images and numpy masks to a fixed square size. This is for single-channel masks."""

    # UNet requires input size to be multiple of 16
    def __init__(self, size):
        """Store the target square size for resizing operations.

        Args:
            size (int): Desired height and width after resizing. Must be multiple of 16.
        """
        self.size = size

    def __call__(self, image, target):
        """Resize inputs while respecting their data types.

        Args:
            image (PIL.Image.Image): Input image to be resized with bilinear interpolation.
            target (numpy.ndarray): Segmentation mask resized with nearest neighbor.

        Returns:
            tuple: `(image, target)` resized to `(size, size)`.
        """
        image = F.resize(
            image,
            (self.size, self.size),
            interpolation=transforms.InterpolationMode.BILINEAR,
        )
        target = cv.resize(target, (self.size, self.size), interpolation=0)
        return image, target


class ChanneledFixResize:
    """Resize images and multi-channel masks to a common square shape."""

    def __init__(self, size):
        """Store the target square size for per-channel resizing.

        Args:
            size (int): Desired height and width after resizing.
        """
        self.size = size

    def __call__(self, image, target):
        """Resize an image and handle 2D or 3D numpy masks appropriately.

        Args:
            image (PIL.Image.Image): Image to be resized with bilinear interpolation.
            target (numpy.ndarray): Mask array that may be single- or multi-channel.

        Returns:
            tuple: `(image, target)` resized to `(size, size)`.
        """
        # Resize image (assumed to be a PIL image) using torchvision transforms
        image = F.resize(
            image,
            (self.size, self.size),
            interpolation=transforms.InterpolationMode.BILINEAR,
        )

        # Resize target:
        # If target is a multi-channel numpy array with shape (C, H, W)
        if isinstance(target, np.ndarray) and len(target.shape) == 3:
            channels = []
            for c in range(target.shape[0]):
                # Use nearest neighbor interpolation for segmentation masks
                resized_channel = cv.resize(
                    target[c], (self.size, self.size), interpolation=cv.INTER_NEAREST
                )
                channels.append(resized_channel)
            target_resized = np.stack(channels, axis=0)
        else:
            # Otherwise assume target is a 2D numpy array
            target_resized = cv.resize(
                target, (self.size, self.size), interpolation=cv.INTER_NEAREST
            )

        return image, target_resized


class ToTensor:
    """Transform the image to tensor. Scale the image to [0,1] float32.
    Transform the mask to tensor.
    """

    def __call__(self, image, target):
        """Convert the image and mask to PyTorch tensors.

        Args:
            image (PIL.Image.Image): Input image expected by torchvision's `ToTensor`.
            target (numpy.ndarray or PIL.Image.Image): Segmentation mask to convert.

        Returns:
            tuple: `(torch.Tensor, torch.Tensor)` ready for training.
        """
        image = transforms.ToTensor()(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class PILToTensor:
    """Transform the image to tensor. Keep raw type."""

    def __call__(self, image, target):
        """Convert the image to a tensor without scaling pixel intensities."""
        image = F.pil_to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class Normalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        """Store channel statistics used to normalize image tensors.

        Args:
            mean (tuple): Per-channel mean values.
            std (tuple): Per-channel standard deviations.
        """
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        """Normalize the image tensor and leave the target unchanged."""
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
