# 🩻 MedSegX - Open-World Medical Image Segmentation Foundation Model and Database

### A generalist foundation model and database for open-world medical image segmentation

Siqi Zhang*, Qizhe Zhang*, Shanghang Zhang*†, Xiaohong Liu*, Jingkun Yue*, Ming Lu, Huihuan Xu, Jiaxin Yao, Xiaobao Wei, Jiajun Cao, Xiang Zhang, Ming Gao, Jun Shen, Yichang Hao, Yinkui Wang, Xingcai Zhang, Song Wu, Ping Zhang, Shuguang Cui & Guangyu Wang†

[Nature Biomedical Engineering (2025)](https://www.nature.com/articles/s41551-025-01497-3): https://www.nature.com/articles/s41551-025-01497-3

*Corresponding Authors: Guangyu Wang, Shanghang Zhang.*

#### Abstract

<p align="justify">
Vision foundation models have demonstrated vast potential in achieving generalist medical segmentation capability, providing a versatile, task-agnostic solution through a single model. However, current generalist models involve simple pre-training on various medical data containing irrelevant information, often resulting in the negative transfer phenomenon and degenerated performance. Furthermore, the practical applicability of foundation models across diverse open-world scenarios, especially in out-of-distribution (OOD) settings, has not been extensively evaluated. Here we construct a publicly accessible database, MedSegDB, based on a tree-structured hierarchy and annotated from 129 public medical segmentation repositories and 5 in-house datasets. We further propose a Generalist Medical Segmentation model (MedSegX), a vision foundation model trained with a model-agnostic Contextual Mixture of Adapter Experts (ConMoAE) for open-world segmentation. We conduct a comprehensive evaluation of MedSegX across a range of medical segmentation tasks. Experimental results indicate that MedSegX achieves state-of-the-art performance across various modalities and organ systems in in-distribution (ID) settings. In OOD and real-world clinical settings, MedSegX consistently maintains its performance in both zero-shot and data-efficient generalization, outperforming other foundation models.
</p>

![overview](assets/overview.webp)

## 📰 News

🔥 **[2025/09/05]** Our paper is available [online](https://www.nature.com/articles/s41551-025-01497-3)!

🔥 **[2025/08/05]** Our paper is accepted by Nature Biomedical Engineering!

## ⚙️ Setup

### 🏝️ Environment

1. Clone this repository.
```bash
git clone https://github.com/MedSegX/MedSegX-code.git MedSegX
cd MedSegX
```

2. Create a conda environment.
```bash
conda create -n medsegx python=3.10 -y
conda activate medsegx
```

3. Install necessary packages.
```bash
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

### 📦️ Model

1. Download base model checkpoints from [SAM](https://github.com/facebookresearch/segment-anything#model-checkpoints):

| Model | Backbone | Checkpoint |
|----------|:----------:|:-----------:|
| SAM | ViT-B | [sam-vit-b-01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) |
| SAM | ViT-L | [sam-vit-l-0b3195.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) |
| SAM | ViT-H | [sam-vit-h-4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) |

2. Download our MedSegX checkpoint from [Google Drive](https://drive.google.com/file/d/1_lTLYQ1p0Et8GGgUNJWzONVSc6RwbRGe/view?usp=share_link).

The checkpoints should be placed in the `./playground` directory, with the following structure:
```
playground/
├── SAM/
│   ├── sam_vit_b_01ec64.pth
│   ├── sam_vit_l_0b3195.pth
│   └── sam_vit_h_4b8939.pth
└── MedSegX/
    └── medsegx_vit_b.pth
```

### 📊 Data

We provide an example dataset to enable users to easily and quickly perform inference with our MedSegX. You can download the example data from [Google Drive](https://drive.google.com/file/d/1MqI5m-lrRMqQghhbiXAdcgHFRLEuNtPN/view?usp=share_link).

The pre-training, ID and OOD datasets from our MedSegDB database are curated from open-source datasets and can be accessed via the weblinks provided in Supplementary Table 1 (see Supplementary Information). Among these, the data in MedSegDB that permit redistribution are available on [HuggingFace](https://huggingface.co/datasets/medicalai/MedSegDB), in which the data is fully preprocessed and can be used out of the box. No further preprocessing is needed before model training or inference.

The datasets should also be placed in the `./playground` directory. 

When using the data for pre-training and ID evaluation, please organize them in the following structure:
```
playground/
└── MedSegDB/
    └── ID/
        ├── dataset1/
        │   ├── modality_task1/
        │   │   ├── npy_imgs/                   # preprocessed images in .npy format
        │   │   └── npy_gts/                    # corresponding ground truth labels
        │   ├── modality_task2/
        │   │   ├── npy_imgs/
        │   │   └── npy_gts/
        │   └── ...                             # other modality-task pairs
        ├── dataset2/
        │   ├── modality_task1/
        │   │   ├── sequence1/                  # if multiple sequences exist
        │   │   │   ├── npy_imgs/
        │   │   │   └── npy_gts/
        │   │   ├── sequence2/
        │   │   │   ├── npy_imgs/
        │   │   │   └── npy_gts/
        │   │   └── ...                         # other sequences if available
        │   └── ...                             # other modality-task pairs
        └── ...                                 # other datasets
```

For example,
```
playground/
└── MedSegDB/
    └── ID/
        ├── ACDC/
        │   ├── MRI_LeftVentricle/
        │   │   ├── npy_imgs/
        │   │   └── npy_gts/
        │   ├── MRI_MitralValve/
        │   │   ├── npy_imgs/
        │   │   └── npy_gts/
        │   └── ...                             # other modality-task pairs
        ├── BraTS2020/
        │   ├── MRI_BrainCoreTumor/
        │   │   ├── FLAIR/
        │   │   │   ├── npy_imgs/
        │   │   │   └── npy_gts/
        │   │   ├── T1/
        │   │   │   ├── npy_imgs/
        │   │   │   └── npy_gts/
        │   │   └── ...                         # other sequences
        │   └── ...                             # other modality-task pairs
        └── ...                                 # other datasets
```

When using the data for OOD (or real-world) evaluation, please organize them in the following structure:
```
playground/
└── MedSegDB/
    └── OOD/
        ├── cross_site/                         # cross-site shift
        │   ├── modality_task1/
        │   │   ├── dataset1/
        │   │   │   ├── finetune/               # fine-tune data
        │   │   │   │   ├── 005_percent/        # different fine-tune percents
        │   │   │   │   │   ├── npy_imgs/
        │   │   │   │   │   └── npy_gts/
        │   │   │   │   ├── 015_percent/
        │   │   │   │   │   ├── npy_imgs/
        │   │   │   │   │   └── npy_gts/
        │   │   │   │   ├── 025_percent/
        │   │   │   │   │   ├── npy_imgs/
        │   │   │   │   │   └── npy_gts/
        │   │   │   │   ├── 050_percent/
        │   │   │   │   │   ├── npy_imgs/
        │   │   │   │   │   └── npy_gts/
        │   │   │   │   └── 100_percent/
        │   │   │   │       ├── npy_imgs/
        │   │   │   │       └── npy_gts/
        │   │   │   └── inference/              # inference data
        │   │   │       ├── npy_imgs/
        │   │   │       └── npy_gts/
        │   │   ├── dataset2/
        │   │   └── ...                         # other datasets
        │   ├── modality_task2/
        │   │   ├── dataset1/
        │   │   │   ├── sequence1/              # if multiple sequences exist
        │   │   │   │   ├── finetune/
        │   │   │   │   │   ├── 005_percent/
        │   │   │   │   │   │   ├── npy_imgs/
        │   │   │   │   │   │   └── npy_gts/
        │   │   │   │   │   ├── 015_percent/
        │   │   │   │   │   │   ├── npy_imgs/
        │   │   │   │   │   │   └── npy_gts/
        │   │   │   │   │   ├── 025_percent/
        │   │   │   │   │   │   ├── npy_imgs/
        │   │   │   │   │   │   └── npy_gts/
        │   │   │   │   │   ├── 050_percent/
        │   │   │   │   │   │   ├── npy_imgs/
        │   │   │   │   │   │   └── npy_gts/
        │   │   │   │   │   └── 100_percent/
        │   │   │   │   │       ├── npy_imgs/
        │   │   │   │   │       └── npy_gts/
        │   │   │   │   └── inference/
        │   │   │   │       ├── npy_imgs/
        │   │   │   │       └── npy_gts/
        │   │   │   ├── sequence2/
        │   │   │   └── ...                     # other sequences if available
        │   │   ├── dataset2/
        │   │   └── ...                         # other datasets
        │   └── ...                             # other modality-task pairs
        └── cross_task/                         # cross-task shift
```

For example,
```
playground/
└── MedSegDB/
    └── OOD/
        ├── cross_site/
        │   ├── CT_Liver/
        │   │   ├── SLIVER07/
        │   │   │   ├── finetune/
        │   │   │   │   ├── 005_percent/
        │   │   │   │   │   ├── npy_imgs/
        │   │   │   │   │   └── npy_gts/
        │   │   │   │   ├── 015_percent/
        │   │   │   │   │   ├── npy_imgs/
        │   │   │   │   │   └── npy_gts/
        │   │   │   │   ├── 025_percent/
        │   │   │   │   │   ├── npy_imgs/
        │   │   │   │   │   └── npy_gts/
        │   │   │   │   ├── 050_percent/
        │   │   │   │   │   ├── npy_imgs/
        │   │   │   │   │   └── npy_gts/
        │   │   │   │   └── 100_percent/
        │   │   │   │       ├── npy_imgs/
        │   │   │   │       └── npy_gts/
        │   │   │   └── inference/
        │   │   │       ├── npy_imgs/
        │   │   │       └── npy_gts/
        │   │   └── ...                         # other datasets
        │   ├── MRI_Spleen/
        │   │   ├── CHAOS/
        │   │   │   ├── T2W/
        │   │   │   │   ├── finetune/
        │   │   │   │   │   ├── 005_percent/
        │   │   │   │   │   │   ├── npy_imgs/
        │   │   │   │   │   │   └── npy_gts/
        │   │   │   │   │   ├── 015_percent/
        │   │   │   │   │   │   ├── npy_imgs/
        │   │   │   │   │   │   └── npy_gts/
        │   │   │   │   │   ├── 025_percent/
        │   │   │   │   │   │   ├── npy_imgs/
        │   │   │   │   │   │   └── npy_gts/
        │   │   │   │   │   ├── 050_percent/
        │   │   │   │   │   │   ├── npy_imgs/
        │   │   │   │   │   │   └── npy_gts/
        │   │   │   │   │   └── 100_percent/
        │   │   │   │   │       ├── npy_imgs/
        │   │   │   │   │       └── npy_gts/
        │   │   │   │   └── inference/
        │   │   │   │       ├── npy_imgs/
        │   │   │   │       └── npy_gts/
        │   │   │   └── ...                     # other sequences
        │   │   └── ...                         # other datasets
        │   └── ...                             # other modality-task pairs
        └── cross_task/
```

## 🛠️ Usage

### 🚀 Training

After preparing the whole MedSegDB, you can train MedSegX using the following command:
```bash
python pretrain.py
```

This command trains MedSegX on 8 GPUs by default, with the batch size of 1024, requiring at least 40GB memory for each GPU. You can set the `device_ids` and `batch_size` arguments for different machine. For example, if you want to train MedSegX on a single GPU, you can use the following command:
```bash
python pretrain.py --device_ids 0 --batch_size 128 --lr 1e-4
```

After pre-training, you can further fine-tune MedSegX on the downstream tasks by using the following command:
```bash
python finetune.py
```

Similarly, 8 GPUs are used for training by default. You can adjust the same arguments for different machines.

### 🧪 Evaluation

For internal evalution, you can run the following command:
```bash
python evaluate_internal.py
```

For external evalution with cross site shift, you can run the following command:
```bash
python evaluate_external.py --shift_type cross_site
```

For external evalution with cross task shift, you can run the following command:
```bash
python evaluate_external.py --shift_type cross_task
```

After running the evaluation commands, the results will be saved in the `./playground/MedSegX/example` folder.

You can download our evalution results from [Google Drive](https://drive.google.com/file/d/1I1kmHtRGAs2wUiuvFgPDte9Mo_y3X1EC/view?usp=share_link) for double checking.

## 🔖 Citation
If you find MedSegX useful for your research and applications, please cite this work:
```
Zhang, S., Zhang, Q., Zhang, S. et al. A generalist foundation model and database for open-world medical image segmentation. Nat. Biomed. Eng (2025). https://doi.org/10.1038/s41551-025-01497-3
```

## 🎟️ License

This project is released under the [Apache 2.0 license](LICENSE).

## 🏅 Acknowledgement

We appreciate the open-source efforts of [SAM](https://github.com/facebookresearch/segment-anything) and [MedSAM](https://github.com/bowang-lab/MedSAM) teams.
