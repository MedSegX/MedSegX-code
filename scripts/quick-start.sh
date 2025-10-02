#!/bin/bash

python evaluate_internal.py \
    --checkpoint ./playground/SAM \
    --model_type vit_b \
    --model_weight ./playground/MedSegX/medsegx_vit_b.pth \
    --data_path ./playground/MedSegDB-example/eval/ID \
    --metric dsc hd \
    --device_ids 0 \
    --batch_size 32

python evaluate_external.py \
    --checkpoint ./playground/SAM \
    --model_type vit_b \
    --model_weight ./playground/MedSegX/medsegx_vit_b.pth \
    --data_path ./playground/MedSegDB-example/eval/OOD \
    --shift_type cross_site \
    --metric dsc hd \
    --device_ids 0 \
    --batch_size 32

python evaluate_external.py \
    --checkpoint ./playground/SAM \
    --model_type vit_b \
    --model_weight ./playground/MedSegX/medsegx_vit_b.pth \
    --data_path ./playground/MedSegDB-example/eval/OOD \
    --shift_type cross_task \
    --metric dsc hd \
    --device_ids 0 \
    --batch_size 32

python evaluate_external.py \
    --checkpoint ./playground/SAM \
    --model_type vit_b \
    --model_weight ./playground/MedSegX/medsegx_vit_b.pth \
    --data_path ./playground/MedSegDB-example/eval/RealWorld \
    --shift_type cross_site \
    --metric dsc hd \
    --device_ids 0 \
    --batch_size 32

python evaluate_external.py \
    --checkpoint ./playground/SAM \
    --model_type vit_b \
    --model_weight ./playground/MedSegX/medsegx_vit_b.pth \
    --data_path ./playground/MedSegDB-example/eval/RealWorld \
    --shift_type cross_task \
    --metric dsc hd \
    --device_ids 0 \
    --batch_size 32