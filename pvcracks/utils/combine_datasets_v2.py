"""
This is a script to combine many different datasets into one when downloading from Supervisely.

Note: if you download from the DuraMAT DataHub, you will not need this.
"""

import os
import shutil

# Path to the directory where the original datasets are stored
dupont = "/Users/ojas/Desktop/saj/SANDIA/pvcracks_data/2025_12_8_CWRU-Dupont_mono"
sunedison = "/Users/ojas/Desktop/saj/SANDIA/pvcracks_data/2025_12_8_CWRU-SunEdison_mono"
lbnl = "/Users/ojas/Desktop/saj/SANDIA/pvcracks_data/2025_12_8_LBNL"
asu = "/Users/ojas/Desktop/saj/SANDIA/pvcracks_data/2025_12_8_ASU"
paths = [dupont, sunedison, lbnl, asu]

# Path to the directory where the combined dataset will be stored
combined = "/Users/ojas/Desktop/saj/SANDIA/pvcracks_data/2025_12_8_Channeled_Combined"

combined_ann = os.path.join(combined, "ann", "json")
combined_img = os.path.join(combined, "img", "original")


def get_single_subfolder(parent):
    subfolders = [
        os.path.join(parent, d)
        for d in os.listdir(parent)
        if os.path.isdir(os.path.join(parent, d))
    ]
    if len(subfolders) != 1:
        raise RuntimeError(f"{parent} does not contain exactly one subfolder")
    return subfolders[0]


# Copy the files from the original datasets to the combined dataset directory
for path in paths:
    root = get_single_subfolder(path)

    ann_src = os.path.join(root, "ann")
    img_src = os.path.join(root, "img")

    if os.path.isdir(ann_src):
        for root_dir, _, files in os.walk(ann_src):
            for file in files:
                print(os.path.join(root_dir, file))
                shutil.copy(os.path.join(root_dir, file), combined_ann)

    if os.path.isdir(img_src):
        for root_dir, _, files in os.walk(img_src):
            for file in files:
                print(os.path.join(root_dir, file))
                shutil.copy(os.path.join(root_dir, file), combined_img)

# Check the number of files in the combined dataset directory
num_ann = len(
    [
        name
        for name in os.listdir(combined_ann)
        if os.path.isfile(os.path.join(combined_ann, name))
    ]
)

num_imgs = len(
    [
        name
        for name in os.listdir(combined_img)
        if os.path.isfile(os.path.join(combined_img, name))
    ]
)

print(f"Number of annotation files in the combined dataset: {num_ann}")
print(f"Number of image files in the combined dataset: {num_imgs}")
