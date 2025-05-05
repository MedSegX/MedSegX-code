import os
join = os.path.join

import torch
from torch.utils.data import Dataset
import numpy as np

from .datainfo import *


class GeneralMedSegDB(Dataset):
    def __init__(self, data_root, train=True):
        self.data_root = data_root
        self.train = train
        
        self.file_names = []
        self.task_names = []
        for dataset in sorted(os.listdir(data_root)):
            dataset_dir = join(data_root, dataset)
            for task in sorted(os.listdir(dataset_dir)):
                task_dir = join(dataset_dir, task)
                if "npy_gts" in os.listdir(task_dir):
                    files = sorted(os.listdir(join(task_dir, "npy_gts")))
                    self.file_names += [join(task_dir, "npy_gts", f) for f in files]
                    self.task_names += [task] * len(files)
                else:
                    for sequence in os.listdir(task_dir):
                        sequence_dir = join(task_dir, sequence)
                        files = sorted(os.listdir(join(sequence_dir, "npy_gts")))
                        self.file_names += [join(sequence_dir, "npy_gts", f) for f in files]
                        self.task_names += [task] * len(files)
    
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        img = np.load(self.file_names[index].replace("npy_gts", "npy_imgs")).transpose(2, 0, 1)
        gt = np.load(self.file_names[index])
        
        y_indices, x_indices = np.where(gt > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # add perturbation to bounding box coordinates
        if self.train:
            H, W = gt.shape
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(W, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(H, y_max + np.random.randint(0, 20))
        box = np.array([x_min, y_min, x_max, y_max])
        
        task = self.task_names[index]
        modal = task.split('_')[0]
        organ = ('_').join(task.split('_')[1:])
        
        modal = modal_map[modal_dict[modal]]
        organ = organ.replace('_', '')
        organ = organ.rstrip('0123456789')
        for k, v in organ_level_1_dict.items():
            if organ in v:
                organ_level_1 = organ_level_1_map[k]
                break
        for k, v in organ_level_2_dict.items():
            if organ in v:
                organ_level_2 = organ_level_2_map[k]
                break
        for k, v in organ_level_3_dict.items():
            if organ in v:
                organ_level_3 = organ_level_3_map[k]
                break
        organ_level_4 = task_idx[organ]
        
        data = {
            "img": torch.tensor(img).float(),
            "box": torch.tensor(box).float(),
            "name": self.file_names[index],
            "modal": modal,
            "organ": (organ_level_1, organ_level_2, organ_level_3, organ_level_4),
        }
        return data, torch.tensor(gt[None, :, :]).long()


class TaskMedSegDB(Dataset):
    def __init__(self, data_root, train=True):
        self.data_root = data_root
        self.train = train
        
        files = sorted(os.listdir(join(data_root, "npy_gts")))
        self.file_names = [join(data_root, "npy_gts", f) for f in files]
        
        # self.task = data_root.split("/")[-3]
        if "inference" in data_root:
            self.task = data_root.split('/')[-4]
            if "cross" in self.task:
                self.task = data_root.split("/")[-3]
        elif "percent" in data_root:
            self.task = data_root.split('/')[-5]
            if "cross" in self.task:
                self.task = data_root.split("/")[-4]
        else:
            self.task = data_root.split("/")[-1]
            if "_" not in self.task or "CH" in self.task:
                self.task = data_root.split("/")[-2]
    
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        img = np.load(self.file_names[index].replace("npy_gts", "npy_imgs")).transpose(2, 0, 1)
        gt = np.load(self.file_names[index])
        
        y_indices, x_indices = np.where(gt > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # add perturbation to bounding box coordinates
        if self.train:
            H, W = gt.shape
            x_min = max(0, x_min - np.random.randint(0, 20))
            x_max = min(W, x_max + np.random.randint(0, 20))
            y_min = max(0, y_min - np.random.randint(0, 20))
            y_max = min(H, y_max + np.random.randint(0, 20))
        box = np.array([x_min, y_min, x_max, y_max])
        
        task = self.task
        modal = task.split('_')[0]
        organ = ('_').join(task.split('_')[1:])
        
        modal = modal_map[modal_dict[modal]]
        organ = organ.replace('_', '')
        organ = organ.rstrip('0123456789')
        for k, v in organ_level_1_dict.items():
            if organ in v:
                organ_level_1 = organ_level_1_map[k]
                break
        for k, v in organ_level_2_dict.items():
            if organ in v:
                organ_level_2 = organ_level_2_map[k]
                break
        for k, v in organ_level_3_dict.items():
            if organ in v:
                organ_level_3 = organ_level_3_map[k]
                break
        if organ not in task_idx:
            organ = "LungCancer"
        organ_level_4 = task_idx[organ]
        
        data = {
            "img": torch.tensor(img).float(),
            "box": torch.tensor(box).float(),
            "name": self.file_names[index],
            "modal": modal,
            "organ": (organ_level_1, organ_level_2, organ_level_3, organ_level_4),
        }
        return data, torch.tensor(gt[None, :, :]).long()
