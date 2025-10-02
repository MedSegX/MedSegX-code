#!/bin/bash

python pretrain.py \
    --checkpoint ./playground/SAM \
    --model_type vit_b \
    --data_path ./playground/MedSegDB \
    --device_ids 0 1 2 3 4 5 6 7 \
    --num_epochs 30 \
    --batch_size 1024 \
    --lr 1e-3 \
    --use_amp