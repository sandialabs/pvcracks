"""
We worked on individual datasets at first, and then had to combine them. This moves files from multiple datasets into a single combined dataset directory.

Note: if you download from the DuraMAT DataHub, you will not need this.
"""

import os
import shutil

# Path to the directory where the original datasets are stored
dupont = "/Users/ojas/Desktop/saj/SANDIA/pvcracks_data/Fresh_LBNL/"
sunedison = "/Users/ojas/Desktop/saj/SANDIA/pvcracks_data/Fresh_ASU/"
lbnl = "/Users/ojas/Desktop/saj/SANDIA/pvcracks_data/CWRU_Dupont_Mono/"
asu = "/Users/ojas/Desktop/saj/SANDIA/pvcracks_data/Fresh_CWRU_SunEdison/"
paths = [dupont, sunedison, lbnl, asu]

# Path to the directory where the combined dataset will be stored
combined = "/Users/ojas/Desktop/saj/SANDIA/pvcracks_data/Channeled_Combined_CWRU_LBNL_ASU_No_Empty_RNE_Revise/"

additives = ["img/original/", "ann_json/"]
# note: because the "fresh" datasets were saved in an old file strucutre (ann_json instead of ann/json),
# once this is run we have to manually move over the files to the new structure (ann/json) in the combined folder

# Copy the files from the original datasets to the combined dataset directory
for path in paths:
    for additive in additives:
        for root, dirs, files in os.walk(path + additive):
            for file in files:
                shutil.copy(os.path.join(root, file), combined + additive)

# Check the number of files in the combined dataset directory
num_files = len(
    [
        name
        for name in os.listdir(combined + additive)
        if os.path.isfile(os.path.join(combined + additive, name))
    ]
)
print(f"Number of files in the combined dataset: {num_files}")
