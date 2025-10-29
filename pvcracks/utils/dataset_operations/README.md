# Dataset Operations

These are a set of tasks for manipulating a dataset that we needed to do. You may need them too.

1. `combine_datasets_cwru_lbnl_asu.py`
   1. We worked on individual datasets at first, and then had to combine them. If you download from the DuraMAT DataHub, you will not need this.

2. `remove_empty_channels.ipynb`
   1. As it stands, `get_channeled_masks.ipynb` creates masks with an additional `empty` layer. This will remove them. If you download from the DuraMAT DataHub, you will not need this.

3. `copy_train_val_to_all.py`
   1. Copies data in `ann/train` and `ann/val` to `ann/all`. If you download from the DuraMAT DataHub, you will not need this.