# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Adapted from https://github.com/facebookresearch/segment-anything

from setuptools import setup

setup(
    name="medsegx",
    version="0.1.0",
    description='A generalist foundation model and database for open-world medical image segmentation',
    author="Qizhe Zhang",
    python_requires=">=3.10",
    install_requires=["matplotlib", "monai", "numpy", "opencv-python", "pandas", "scikit-image", "scipy", "tqdm"],
)
