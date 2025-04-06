# Generating masks for formatted images and masks on Supervisely

1. `chmod +x create_directories.sh` and then `./create_directories.sh <root folder>` to create the required file structure.

2. Download the dataset from Supervisely:
   1. Download in the Supervisely format, select "json annotations + images", and don't select "fix image extension..".
   2. Leave advanced settings as default.
   3. Place the image files to `/img/orignal/`. Place `.json` files in `/ann_json/`.

3. Convert `.json` mask to a `.npy` mask using `ojas_get_masks.ipynb`
   1. This notebook takes annotations in the form of bitmaps, and transforms them into a five-layer tensor of 1s and 0s before squashing the mask into a 2 dimensional array of integers in [0, 1, 2, 3, 4]
   2. The order in which layers are stacked determines priority in the final 2D target. The final target can only have one value for each pixel location.
   3. Masks with cracks prioritized are saved into `/ann_npy_split_cracks/original/`
   4. Masks with busbars prioritized are saved into `/ann_npy_split_busbars/original`

4. Flip, mirror, and rotate `.jpg` images and target `.npy` masks, with `ojas_flip_rotate_mirror.ipynb`
   1. There should be five subfolders in `/img/`, `/ann_npy_split_busbars/`, and `/ann_npy_split_cracks/`: 
    - `original`, 
    - `mirrored_x`, 
    - `mirrored_y`, 
    - `mirrored_xy`, 
    - and `all`. 
   2. Arrays that get flipped across the y-axis are saved in `mirrored_y` with a prefix "my_" on the file name. Similar for "mx_" in `mirrored_xy` and "mxy_" in `mirrored_xy`. 
   3. Everything also gets saved in `all`, which is ultimately used to split the files into training and testing sets. The prefixes distinguish the files when they are put in `all`.

5. Afterwards, before training, run `../training/ojas_split_train_val.ipynb` to split the files into training and testing sets.

The final `tree -d` directory structure should look like this:
```
├── all
├── ann # .npy masks
│   ├── train
│   └── val
├── ann_json # .json masks
├── ann_npy_split_busbars # .npy masks
│   ├── all
│   ├── mirrored_x
│   ├── mirrored_xy
│   ├── mirrored_y
│   └── original
├── ann_npy_split_cracks # .npy masks
│   ├── all
│   ├── mirrored_x
│   ├── mirrored_xy
│   ├── mirrored_y
│   └── original
├── checkpoints
│   ├── dm_checkpoints2
│   ├── dm_checkpoints5
│   └── dm_checkpoints6
├── img # .jpg images
│   ├── train
│   ├── val
│   ├── all
│   ├── mirrored_x
│   ├── mirrored_xy
│   ├── mirrored_y
│   └── original
```


6. Finally, run `ojas_train.ipynb`.

---

To remove `.DS_Store` files on Mac, run this in `/pvcracks`:

```
find ./ -type f -name ".DS_Store" -exec rm -f {} +
```

----

With channeled:

1. `create_channeled_directories.sh`
2. Run `ojas_get_channeled_masks.ipynb` to make new masks, saved in  `ann/channeled`
3. Copy from `{original}/img/original/` -> `{channeled}/img/original`
   1. something like: `cp -r {original}/img/original/` -> `{channeled}/img/`
4. Run `ojas_channeled_masks_flip_rotate_mirror_and_split_train_val.ipynb`
5. Run `ojas_train_channeled_unet.ipynb`
6. Profit