#!/bin/bash

# Check if the root folder is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <root_folder>"
    exit 1
fi

ROOT_FOLDER=$1

# Create the directory structure under the specified root folder
mkdir -p "$ROOT_FOLDER/ann/all"
mkdir -p "$ROOT_FOLDER/ann/channeled"
mkdir -p "$ROOT_FOLDER/ann/mirrored_x"
mkdir -p "$ROOT_FOLDER/ann/mirrored_y"
mkdir -p "$ROOT_FOLDER/ann/mirrored_xy"
mkdir -p "$ROOT_FOLDER/ann/train"
mkdir -p "$ROOT_FOLDER/ann/val"
mkdir -p "$ROOT_FOLDER/ann/json"

mkdir -p "$ROOT_FOLDER/checkpoints/"

mkdir -p "$ROOT_FOLDER/img/all"
mkdir -p "$ROOT_FOLDER/img/original"
mkdir -p "$ROOT_FOLDER/img/mirrored_x"
mkdir -p "$ROOT_FOLDER/img/mirrored_xy"
mkdir -p "$ROOT_FOLDER/img/mirrored_y"
mkdir -p "$ROOT_FOLDER/img/train"
mkdir -p "$ROOT_FOLDER/img/val"

echo "Directory structure created successfully in $ROOT_FOLDER!"