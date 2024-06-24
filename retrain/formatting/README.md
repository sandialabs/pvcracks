**Starting from an image of a mini-module**
1. Use [transform_contour_rounded_minimodule.ipynb]() to straighten and crop image
    1. This approach uses the MaskModule class from pv-vision.transform_crop.solarmodule
    2. Save the straightened images in an intermediate folder
2. Use [extract_cells.ipynb]() to slice individual cells out of the straightened images
    1. Approach from Ben Pierce, uses skimage.measure.regionprops
    2. Save images of individual cells into ```/projects/wg-psel-ml/EL_images/eccoope/img_cropped/original```

**Starting from an image of a cell** \
Use [transform_contour_rounded_cells.ipynb]() to straighten and crop image. Save images into ```/projects/wg-psel-ml/EL_images/eccoope/img_cropped/original```.

**Now, download images locally and make Supevisely masks for them.** Classes are "crack", "busbar", and "dark". Once you make the mask, download it from the Supervisely platform as a .json file. Upload the file back to ```/projects/wg-psel-ml/EL_images/eccoope/ann_json```.

**Convert the .json mask to a .npy mask.** using [get_masks.ipynb]
1. This notebook takes annotations in the form of bitmaps, and transforms them into a five-layer tensor of 1s and 0s before squashing the mask into a 2 dimensional array of integers in [0, 1, 2, 3, 4]
2. The order in which layers are stacked determines priority in the final 2d target. The final target can only have one value for each pixel location.
3. Save masks with cracks prioritized into ```/projects/wg-psel-ml/EL_images/eccoope/ann_npy_split_cracks```
4. Save masks with busbars prioritized into ```/projects/wg-psel-ml/EL_images/eccoope/ann_npy_split_busbars```

**Flip, mirror, and rotate .jpg images and target .npy masks**
Using [flip_rotate_mirror.ipynb](). There should be five subfolders in ```img_cropped``` and ```ann_npy_split_busbars``` / ```ann_npy_split_cracks``` - ```original```, ```mirrored_x```, ```mirrored_y```, ```mirrored_xy```, and ```all```. Arrays that get flipped across the y-axis are saved in ```mirrored_y``` with a prefix "my_" on the file name. Similar for "mx_" in ```mirrored_xy``` and "mxy_" in ```mirrored_xy```. Everything also gets saved in ```all```, which is ultimately used to split the files into training and testing sets. The prefixes distinguish the files when they are put in ```all```.