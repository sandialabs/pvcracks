# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 12:11:11 2023

@author: jlbraid
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.transform is not None:
            image = self.data[idx]
            print(f"Shape of image from data: {image.shape}")

            # image = image.numpy()
            # print(f"Shape of image after conversion to numpy: {image.shape}")

            # image = torch.from_numpy(image).float()
            # print(f"Shape of image after conversion to tensor: {image.shape}")
            
            image = self.transform(image)
            print(f"Shape of image after transformation: {image.shape}")

            image = image.squeeze()
        
            image = image.unsqueeze(0)  # Add channel dimension (C, H, W)
            print(f"Shape of image after adding channel dimension: {image.shape}")

        else:
            image = self.data[idx]
            print(f"Shape of image from data: {image.shape}")
        
            image = torch.from_numpy(image).float()
            print(f"Shape of image after conversion to tensor: {image.shape}")
        
            image = image.squeeze()
        
            image = image.unsqueeze(0)  # Add channel dimension (C, H, W)
            print(f"Shape of image after adding channel dimension: {image.shape}")
        
        return image


class FixedRotation:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return transforms.functional.rotate(x, self.angle)

class FixedHorizontalFlip:
    def __call__(self, x):
        return torch.flip(x, [-1])

class FixedVerticalFlip:
    def __call__(self, x):
        return torch.flip(x, [-2])
