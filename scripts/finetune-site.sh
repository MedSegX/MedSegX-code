#!/bin/bash

python finetune.py \
    --checkpoint ./playground/SAM \
    --model_type vit_b \
    --data_path ./playground/MedSegDB/eval/OOD \
    --shift_type cross_site \
    --device_ids 0 1 2 3 4 5 6 7 \
    --num_epochs 30 \
    --batch_size 64 \
    --validation val \
    --resume ./playground/MedSegX/medsegx_vit_b.pth \
    --lr 5e-5 \
    --use_amp