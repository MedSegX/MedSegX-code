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

## Usage
You can download the [model checkpoint](https://drive.google.com/file/d/1dDqHr_W7V-_D7Tra_E5AMNn0-jrGhJdT/view?usp=sharing) and [example data](https://drive.google.com/file/d/1k_R7xMYO8IVwm8J4fr83qYqX3-laaQvc/view?usp=share_link) for inference.

## Dataset

MedSegDB was constructed through the preprocessing of publicly available medical image segmentation datasets. 

👉 It is available on [HuggingFace](https://huggingface.co/datasets/medicalai/MedSegDB).

For additional information about the datasets or their licenses, please reach out to the owners.

## Acknowledgement

We appreciate the open-source efforts of [SAM](https://github.com/facebookresearch/segment-anything) team.

