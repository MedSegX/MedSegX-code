# MedSegX
This is the official repository for A generalist foundation model and database for open-world medical image segmentation (MedSegX).

## Train
Training:
```
python pretrain.py
```
Fine-tuning:
```
python finetune.py
```

## Evaluate
ID evaluation:
```
python evaluate_id.py
```
OOD evaluation:
```
python evaluate_ood.py
```

## Dataset

MedSegDB was constructed through the preprocessing of publicly available medical image segmentation datasets. 

👉 It is available on [HuggingFace](https://huggingface.co/datasets/medicalai/MedSegDB).

For additional information about the datasets or their licenses, please reach out to the owners.

## Acknowledgement

We appreciate the open-source efforts of [SAM](https://github.com/facebookresearch/segment-anything) and [MedSAM](https://github.com/bowang-lab/MedSAM) teams.

