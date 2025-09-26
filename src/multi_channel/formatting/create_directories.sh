#!/bin/bash

# Check if the root folder is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <root_folder>"
    exit 1
fi

ROOT_FOLDER=$1

# Create the directory structure under the specified root folder
mkdir -p "$ROOT_FOLDER/all"

mkdir -p "$ROOT_FOLDER/ann/train"
mkdir -p "$ROOT_FOLDER/ann/val"

mkdir -p "$ROOT_FOLDER/ann_json"

mkdir -p "$ROOT_FOLDER/ann_npy_split_busbars/all"
mkdir -p "$ROOT_FOLDER/ann_npy_split_busbars/mirrored_x"
mkdir -p "$ROOT_FOLDER/ann_npy_split_busbars/mirrored_xy"
mkdir -p "$ROOT_FOLDER/ann_npy_split_busbars/mirrored_y"
mkdir -p "$ROOT_FOLDER/ann_npy_split_busbars/original"

mkdir -p "$ROOT_FOLDER/ann_npy_split_cracks/all"
mkdir -p "$ROOT_FOLDER/ann_npy_split_cracks/mirrored_x"
mkdir -p "$ROOT_FOLDER/ann_npy_split_cracks/mirrored_xy"
mkdir -p "$ROOT_FOLDER/ann_npy_split_cracks/mirrored_y"
mkdir -p "$ROOT_FOLDER/ann_npy_split_cracks/original"

mkdir -p "$ROOT_FOLDER/checkpoints/"

mkdir -p "$ROOT_FOLDER/img/train"
mkdir -p "$ROOT_FOLDER/img/val"
mkdir -p "$ROOT_FOLDER/img/all"
mkdir -p "$ROOT_FOLDER/img/mirrored_x"
mkdir -p "$ROOT_FOLDER/img/mirrored_xy"
mkdir -p "$ROOT_FOLDER/img/mirrored_y"
mkdir -p "$ROOT_FOLDER/img/original"

echo "Directory structure created successfully in $ROOT_FOLDER!"