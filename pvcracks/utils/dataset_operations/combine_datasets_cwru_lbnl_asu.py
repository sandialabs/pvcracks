# copy files in python

import os
import shutil

# Path to the directory where the original datasets are stored
dupont = '/Users/ojas/Desktop/saj/SANDIA/pvcracks_data/CWRU_Dupont_Mono/'
sunedison = '/Users/ojas/Desktop/saj/SANDIA/pvcracks_data/CWRU_SunEdison_Mono/'
lbnl = '/Users/ojas/Desktop/saj/SANDIA/pvcracks_data/LBNL_Mono_Cells/'
asu = '/Users/ojas/Desktop/saj/SANDIA/pvcracks_data/ASU_IHDEANE/'

# Path to the directory where the combined dataset will be stored
combined = '/Users/ojas/Desktop/saj/SANDIA/pvcracks_data/Combined_CWRU_LBNL_ASU/'
    
additives = ["ann/train/", "ann/val/", "img/train/", "img/val/"]

# Copy the files from the original datasets to the combined dataset directory
for additive in additives:
    for root, dirs, files in os.walk(dupont + additive):
        for file in files:
            shutil.copy(os.path.join(root, file), combined + additive) 
            
for additive in additives:
    for root, dirs, files in os.walk(sunedison + additive):
        for file in files:
            shutil.copy(os.path.join(root, file), combined + additive)
            
for additive in additives:
    for root, dirs, files in os.walk(lbnl + additive):
        for file in files:
            shutil.copy(os.path.join(root, file), combined + additive)

for additive in additives:
    for root, dirs, files in os.walk(asu + additive):
        for file in files:
            shutil.copy(os.path.join(root, file), combined + additive)
            
# Check the number of files in the combined dataset directory
num_files = len([name for name in os.listdir(combined + additive) 
                 if os.path.isfile(os.path.join(combined + additive, name))])
print(f'Number of files in the combined dataset: {num_files}')
