#!/bin/bash

python evaluate_external.py \
    --checkpoint ./playground/SAM \
    --model_type vit_b \
    --model_weight ./playground/MedSegX/medsegx_vit_b.pth \
    --data_path ./playground/MedSegDB/eval/RealWorld \
    --shift_type cross_task \
    --metric dsc hd \
    --device_ids 0 1 2 3 4 5 6 7 \
    --batch_size 32