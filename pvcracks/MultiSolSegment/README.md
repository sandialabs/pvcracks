# Generating masks for formatted images and masks

Note: this is for development, if you are downloading from Supervisely. **If you are an end-user downloading from the DuraMat DataHub, skip to step 6.**

1. `chmod +x create_channeled_directories.sh` and then `./create_channeled_directories.sh <root folder>` to create the required file structure.

2. Download the data:
   1. Download in the Supervisely format, select "json annotations + images", and don't select "fix image extension..". 
   2. Leave advanced settings as default.
   3. Place the image files in `/img/orignal/`. Place `.json` files in `/ann/json/`.

3. Convert `.json` mask to a `.npy` mask using `get_channeled_masks.ipynb`
   1. This notebook takes annotations in the form of bitmaps, and transforms them into an array of 2D arrays (ie, a 3D array). Each 2D array contains binary per-pixel activation maps for one of the 4 classes.
   2. These masks are saved in `/ann/channeled`.

4. Flip, mirror, and rotate `.jpg` images and target `.npy` masks; and split them into training and validation sets with `channeled_masks_flip_rotate_mirror_and_split_train_val.ipynb`
   1. There are six subfolders in `/img/` and `/ann/`: 
    - `original`, 
    - `mirrored_x`, 
    - `mirrored_y`, 
    - `mirrored_xy`, 
    - `all`,
    - `train`,
    - and `val`.
   2. Arrays that get flipped across the y-axis are saved in `mirrored_y` with a prefix "my_" on the file name. Similar for "mx_" in `mirrored_xy` and "mxy_" in `mirrored_xy`. 
   3. Everything also gets saved in `all`. The prefixes distinguish the files when they are put in `all`.
   4. Finally, the images and masks are taken from their respective `all` folders and split into training and validation sets, in `/train` and `/val`

<!-- 5. Finally, you must run `/utils/dataset_operations/remove_empty_channels.ipynb` to remove an additional layer created during the mask creation process. -->

6. The final `tree -d` directory structure should look like this:

```
.
├── ann 
│   ├── all
│   ├── channeled
│   ├── mirrored_x
│   ├── mirrored_xy
│   ├── mirrored_y
│   ├── train
│   ├── json
│   └── val
├── checkpoints
└── img 
    ├── all
    ├── mirrored_x
    ├── mirrored_xy
    ├── mirrored_y
    ├── original
    ├── train
    └── val
```

7. Finally, you can train a model with `train_channeled_unet.ipynb`.
   1. Note, this is for development. If you are training a final model on a test set, use `train_channeled_wandb_k_fold_final_on_test_set.ipynb.`

---

To remove `.DS_Store` files on Mac, run this in `/pvcracks` and `/pvcracks_data` (or wherever you store your data):

```
find ./ -type f -name ".DS_Store" -exec rm -f {} +
```

Note that the presence of a `.DS_Store` file (and thus the necessity to run the above command) is often signified by the following error when running `train_functions.get_save_dir()`:

`ValueError: invalid literal for int() with base 10: 'e'`