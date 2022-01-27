#!/bin/bash

read -p "gpu: " GPU

OUT_DIR="output"
if [[ ! -d "${OUT_DIR}" ]]; then
    echo "Creating output dir for training : ${OUT_DIR}"
    mkdir ${OUT_DIR}
fi
CUDA_VISIBLE_DEVICES=$GPU python3 tools/main.py --cfg configs/hotels_small_resnet50.yaml
