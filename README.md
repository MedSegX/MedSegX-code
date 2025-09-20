# MedSegX - Open-World Medical Image Segmentation Foundation Model and Database

### A generalist foundation model and database for open-world medical image segmentation

Siqi Zhang*, Qizhe Zhang*, Shanghang Zhang*†, Xiaohong Liu*, Jingkun Yue*, Ming Lu, Huihuan Xu, Jiaxin Yao, Xiaobao Wei, Jiajun Cao, Xiang Zhang, Ming Gao, Jun Shen, Yichang Hao, Yinkui Wang, Xingcai Zhang, Song Wu, Ping Zhang, Shuguang Cui & Guangyu Wang†


[Nature Biomedical Engineering (2025)](https://www.nature.com/articles/s41551-025-01497-3): https://www.nature.com/articles/s41551-025-01497-3

*Corresponding Authors: Guangyu Wang, Shanghang Zhang.*

#### Abstract

Vision foundation models have demonstrated vast potential in achieving generalist medical segmentation capability, providing a versatile, task-agnostic solution through a single model. However, current generalist models involve simple pre-training on various medical data containing irrelevant information, often resulting in the negative transfer phenomenon and degenerated performance. Furthermore, the practical applicability of foundation models across diverse open-world scenarios, especially in out-of-distribution (OOD) settings, has not been extensively evaluated. Here we construct a publicly accessible database, MedSegDB, based on a tree-structured hierarchy and annotated from 129 public medical segmentation repositories and 5 in-house datasets. We further propose a Generalist Medical Segmentation model (MedSegX), a vision foundation model trained with a model-agnostic Contextual Mixture of Adapter Experts (ConMoAE) for open-world segmentation. We conduct a comprehensive evaluation of MedSegX across a range of medical segmentation tasks. Experimental results indicate that MedSegX achieves state-of-the-art performance across various modalities and organ systems in in-distribution (ID) settings. In OOD and real-world clinical settings, MedSegX consistently maintains its performance in both zero-shot and data-efficient generalization, outperforming other foundation models.

![overview](assets/overview.webp)

## News

🔥 Our paper is available [online](https://www.nature.com/articles/s41551-025-01497-3)!

🔥 Our paper is accepted by Nature Biomedical Engineering!

## TODO

- [ ] Detailed User Instructions (in 1 week)

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

## Citation
```
Zhang, S., Zhang, Q., Zhang, S. et al. A generalist foundation model and database for open-world medical image segmentation. Nat. Biomed. Eng (2025). https://doi.org/10.1038/s41551-025-01497-3
```

## Acknowledgement

We appreciate the open-source efforts of [SAM](https://github.com/facebookresearch/segment-anything) team.

