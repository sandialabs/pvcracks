"""
Moves files from train and val folders to all folder.

Note: if you download from the DuraMAT DataHub, you will not need this.
"""

# copy files in python

import os
import shutil

# Path to the directory where the original datasets are stored
train = "/Users/ojas/Desktop/saj/SANDIA/pvcracks_data/Channeled_Combined_CWRU_LBNL_ASU_No_Empty/ann/train/"
val = "/Users/ojas/Desktop/saj/SANDIA/pvcracks_data/Channeled_Combined_CWRU_LBNL_ASU_No_Empty/ann/val/"

# Path to the directory where the combined dataset will be stored
all = "/Users/ojas/Desktop/saj/SANDIA/pvcracks_data/Channeled_Combined_CWRU_LBNL_ASU_No_Empty/ann/all"


# Copy the files from the original datasets to the combined dataset directory
for root, dirs, files in os.walk(train):
    for file in files:
        shutil.copy(os.path.join(root, file), all)
        print(f"Copied {file} to {all}")

for root, dirs, files in os.walk(val):
    for file in files:
        shutil.copy(os.path.join(root, file), all)
        print(f"Copied {file} to {all}")
