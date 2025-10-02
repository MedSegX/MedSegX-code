# MedSegDB Data Structure

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
        │   │   │   │   ├── npy_imgs/        
        │   │   │   │   └── npy_gts/
        │   │   │   └── inference/              # inference data
        │   │   │       ├── npy_imgs/
        │   │   │       └── npy_gts/
        │   │   ├── dataset2/
        │   │   └── ...                         # other datasets
        │   ├── modality_task2/
        │   │   ├── dataset1/
        │   │   │   ├── sequence1/              # if multiple sequences exist
        │   │   │   │   ├── finetune/
        │   │   │   │   │   ├── npy_imgs/
        │   │   │   │   │   └── npy_gts/
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
        │   │   │   │   ├── npy_imgs/
        │   │   │   │   └── npy_gts/
        │   │   │   └── inference/
        │   │   │       ├── npy_imgs/
        │   │   │       └── npy_gts/
        │   │   └── ...                         # other datasets
        │   ├── MRI_Spleen/
        │   │   ├── CHAOS/
        │   │   │   ├── T2W/
        │   │   │   │   ├── finetune/
        │   │   │   │   │   ├── npy_imgs/
        │   │   │   │   │   └── npy_gts/
        │   │   │   │   └── inference/
        │   │   │   │       ├── npy_imgs/
        │   │   │   │       └── npy_gts/
        │   │   │   └── ...                     # other sequences
        │   │   └── ...                         # other datasets
        │   └── ...                             # other modality-task pairs
        └── cross_task/
```