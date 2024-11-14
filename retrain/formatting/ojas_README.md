# Generating masks for formatted images and masks on Supervisely

1. Download images from Supervisely, and download masks as `.json`. Upload the image files to the `/img/` subfolder, and `.json` files to `/ann_json/` subfolder.
2. Convert `.json` mask to a `.npy` mask using `get_masks.ipynb`
   1. This notebook takes annotations in the form of bitmaps, and transforms them into a five-layer tensor of 1s and 0s before squashing the mask into a 2 dimensional array of integers in [0, 1, 2, 3, 4]
   2. The order in which layers are stacked determines priority in the final 2D target. The final target can only have one value for each pixel location.
   3. Save masks with cracks prioritized into `/ann_npy_split_cracks/` subfolder
   4. Save masks with busbars prioritized into `/ann_npy_split_busbars/` subfolder
3. Flip, mirror, and rotate `.jpg` images and target `.npy` masks, with `flip_rotate_mirror.ipynb`
   1. There should be five subfolders in `/img/` and `/ann_npy_split_busbars/` / `/ann_npy_split_cracks/`: `original`, `mirrored_x`, `mirrored_y`, `mirrored_xy`, and `all`. 
   2. Arrays that get flipped across the y-axis are saved in `mirrored_y` with a prefix "my_" on the file name. Similar for "mx_" in `mirrored_xy` and "mxy_" in `mirrored_xy`. 
   3. Everything also gets saved in `all`, which is ultimately used to split the files into training and testing sets. The prefixes distinguish the files when they are put in `all`.