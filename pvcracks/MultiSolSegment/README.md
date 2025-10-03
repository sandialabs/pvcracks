# el_img_cracks_ec

Resulting training weights available here: 
https://datahub.duramat.org/dataset/pvcracks-re-trained-pv-vision-model

**Flow**
1. Format images and targets (includes cropping, aligning, converting masks from .json files to .jpg files, mirroring images/masks)
2. Train (includes splitting into training and validation sets, and training)
3. Test (includes plotting activations)

Images and masks stored in `/projects/wg-psel-ml/EL_images/eccoope`
```
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
├── raw_images # .jpg images
├── checkpoints
│   ├── dm_checkpoints2
│   ├── dm_checkpoints5
│   └── dm_checkpoints6
├── img # .jpg images
│   ├── train
│   └── val
└── img_cropped # .jpg images
    ├── all
    ├── mirrored_x
    ├── mirrored_xy
    ├── mirrored_y
    └── original
       
```

