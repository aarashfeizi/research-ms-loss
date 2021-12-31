#!/bin/bash

OUT_DIR="output"
if [[ ! -d "${OUT_DIR}" ]]; then
    echo "Creating output dir for training : ${OUT_DIR}"
    mkdir ${OUT_DIR}
fi
CUDA_VISIBLE_DEVICES=0 python3 tools/main.py --cfg configs/cub_resnet50.yaml
