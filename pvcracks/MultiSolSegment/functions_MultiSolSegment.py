import logging
import os
from pathlib import Path

import torch
import torchvision.transforms.functional as F
from torchvision.datasets.vision import VisionDataset
from imutils.paths import list_images, list_files
from PIL import Image
import cv2 as cv
import numpy as np
from torchvision import transforms

def logger_func(file_name):
    #Logging format, pulled from RTC
    format = "[%(asctime)s:%(filename)s:%(lineno)s:%(levelname)s - %(funcName)20s() ] %(message)s"
    # Build the home_path for the logs placement.
    home_path = os.path.join(Path.home(), 'el_img_cracks_ec',"logs")
    # If the logs directory does not exist create it
    if not(os.path.isdir(home_path)):
        os.makedirs(home_path)
    # Make the path to the log file
    log_path = os.path.join(home_path,file_name)
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO, format=format, filename=log_path)
    logger = logging.getLogger(__name__)
    logger.addHandler(file_handler)
    # If the logger executes inside a lower class or function it will by default
    # Execute for every level of code...so one level down you will get two
    # identical error messages.
    logger.propagate = False

    return logger

def realign_mask(mask):
    # hack to swap masks to correct labels
    idx_nothing = torch.where(mask == 0)
    idx_busbar = torch.where(mask == 4)
    idx_cross = torch.where(mask == 2)
    idx_crack = torch.where(mask == 3)
    idx_dark = torch.where(mask == 1)

    mask[idx_nothing] +=0 #nothing
    mask[idx_busbar] += -3 # dark
    mask[idx_cross] += 1 # cross
    mask[idx_dark] += 3 # busbar
    mask[idx_crack] += -1 # crack

    return mask

class SolarDataset(VisionDataset):
    """A dataset directly read images and masks from folder.    
    """
    def __init__(self, 
                 root, 
                 image_folder, 
                 mask_folder,
                 transforms,
                 mode = "train",
                 random_seed=42):
        super().__init__(root, transforms)
        
        self.image_path = Path(self.root) / image_folder
        self.mask_path = Path(self.root) / mask_folder

        if not os.path.exists(self.image_path):
            raise OSError(f"{self.image_path} not found.")

        if not os.path.exists(self.mask_path):
            raise OSError(f"{self.mask_path} not found.")
            
            
        self.image_list = sorted([c for c in list(list_files(self.image_path)) if '.ipynb_checkpoints' not in c])
        self.mask_list = sorted([c for c in list(list_files(self.mask_path)) if '.ipynb_checkpoints' not in c])
        self.image_list = np.array(self.image_list)
        self.mask_list = np.array(self.mask_list)
        
        np.random.seed(random_seed)
        index = np.arange(len(self.image_list))
        np.random.shuffle(index)
        self.image_list = self.image_list[index]
        self.mask_list = self.mask_list[index]
        
    def __len__(self):
        return len(self.image_list)

    def __getname__(self, index):
        image_name = os.path.splitext(os.path.split(self.image_list[index])[-1])[0]
        mask_name = os.path.splitext(os.path.split(self.mask_list[index])[-1])[0]

        if image_name == mask_name:
            return image_name
        else:
            return False
    
    def __getraw__(self, index):
        if not self.__getname__(index):
            raise ValueError("{}: Image doesn't match with mask".format(os.path.split(self.image_list[index])[-1]))
        image = Image.open(self.image_list[index])
        mask = np.load(self.mask_list[index])

        return image, mask

    def __getitem__(self, index):
        image, mask = self.__getraw__(index)
        image, mask = self.transforms(image, mask)

        return image, mask
    
class Compose:
    def __init__(self, transforms):
        """
        transforms: a list of transform
        """
        self.transforms = transforms
    
    def __call__(self, image, target):
        """
        image: input image
        target: input mask
        """
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

# MODIFIED FOR NUMPY ARRAY INPUT
class FixResize:
    # UNet requires input size to be multiple of 16
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.resize(image, (self.size, self.size), interpolation=transforms.InterpolationMode.BILINEAR)
        target = cv.resize(target, (self.size, self.size), interpolation=0)
        return image, target

class ToTensor:
    """Transform the image to tensor. Scale the image to [0,1] float32.
    Transform the mask to tensor.
    """
    def __call__(self, image, target):
        image = transforms.ToTensor()(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target

class PILToTensor:
    """Transform the image to tensor. Keep raw type."""
    def __call__(self, image, target):
        image = F.pil_to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target

class Normalize:
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std
    
    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target