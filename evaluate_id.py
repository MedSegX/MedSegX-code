# -*- coding: utf-8 -*-
"""
MedSedX ID evaluating script
"""

# setup environment
import argparse
import os
join = os.path.join
import random
import time

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
import numpy as np
import pandas as pd
from tqdm import tqdm

from segment_anything import sam_model_registry, sam_model_checkpoint
from segment_anything.utils.transforms import ResizeLongestSide
from model import *
from data.dataset import TaskMedSegDB
from utils.metric import SegmentMetrics

# set seeds
seed = 2023
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.empty_cache()
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# set up parser
parser = argparse.ArgumentParser("MedSedX ID evaluating", add_help=False)
parser.add_argument("--model_type", type=str, default="vit_b")
parser.add_argument("--task_name", type=str, default="MedSegX_256")
parser.add_argument("--method", type=str, default="medsegx")
parser.add_argument("--bottleneck_dim", type=int, default=16)
parser.add_argument("--embedding_dim", type=int, default=16)
parser.add_argument("--expert_num", type=int, default=4)
parser.add_argument("--checkpoint", type=str, default="/path/to/SAM")
parser.add_argument("--data_path", type=str, default="/path/to/MedSegDB")
parser.add_argument("--metric", type=str, default=["dsc", "nsd", "hd"], nargs='+',
                    help="evaluation metrics (e.g dsc, nsd, hd)")
parser.add_argument("--data_dim", type=str, default="2D")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--device_ids", type=int, default=[0,1,2,3,4,5,6,7], nargs='+',
                    help="device ids assignment (e.g 0 1 2 3)")
parser.add_argument("--work_dir", type=str, default="./work_dir")
# test
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=32)
parser.add_argument("--resume", type=str, default=None, 
                    help="Resuming training from checkpoint")
parser.add_argument("--use_amp", action="store_true", default=False, 
                    help="Use amp")


def evaluate(model, metric, dataloader, img_size, img_transform, box_transform,
             dataset, task, sequence=None, result_total=None, meta=None, args=None):
    model.eval()
    device = torch.device(args.device)
    
    result_task = {}
    for m in args.metric:
        result_task[m] = []
    
    pbar = tqdm(dataloader)
    if sequence is not None:
        pbar.set_description(f"Evaluating - {dataset} {task} {sequence}")
    else:
        pbar.set_description(f"Evaluating - {dataset} {task}")
    with torch.no_grad():
        for data, label in pbar:
            if data["img"].shape[-1] != img_size:
                data["box"] = box_transform.apply_boxes_torch((data["box"].reshape(-1, 2, 2)), 
                                                                data["img"].shape[-2:]).reshape(-1, 4)
                data["img"] = img_transform(data["img"])
            data["img"] = data["img"].to(device, non_blocking=True)
            data["box"] = data["box"].to(device, non_blocking=True)
            label = label.to(device, non_blocking=True, dtype=torch.bool)
            
            if args.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    mask_pred = model(data)
            else:
                mask_pred = model(data)
            
            if mask_pred.shape[-1] != label.shape[-1]:
                mask_pred = F.interpolate(mask_pred, size=label.shape[-1], mode="bilinear", antialias=True)
            mask_prob = torch.sigmoid(mask_pred)
            mask = (mask_prob > 0.5).bool()
            
            result_list = {}
            metric_dict = {}
            metric_list = {m: [] for m in args.metric}
            for m in args.metric:
                result_list[m] = []
            for idx in range(model.module.sam.mask_decoder.num_multimask_outputs):
                result_batch = metric(mask[:, idx].unsqueeze(1), label)
                for m in args.metric:
                    result_list[m].append(result_batch[m])
            
            dsc, max_idx = torch.stack(result_list["dsc"], dim=0).max(dim=0)
            for m in args.metric:
                if m == "dsc":
                    result = dsc
                else:
                    result = torch.stack(result_list[m], dim=0)
                    result = result[max_idx, torch.arange(result.shape[1])]
                metric_dict[m] = result.mean().item()
                metric_list[m].append(result)
                result_task[m].append(result)
                result_total[m].append(result)
            
            pbar.set_postfix(metric_dict)
            metric_list = {m: torch.cat(v) for m, v in metric_list.items()}
            for idx, name in enumerate(data["name"]):
                file = name.replace(f"/path/to/MedSegDB/{args.data_dim}/", "").replace("npy_gts", "npy_imgs")
                meta["File"].append(file)
                for m in args.metric:
                    meta[m.upper()].append(metric_list[m][idx].item())
    
    result_task = {k: torch.cat(v).mean().item() for k, v in result_task.items()}
    return result_task


def main(args):
    device = torch.device(args.device)
    
    checkpoint = join(args.checkpoint, sam_model_checkpoint[args.model_type])
    sam_model = sam_model_registry[args.model_type](image_size=256, keep_resolution=True, checkpoint=checkpoint)
    if args.method == "medsam":
        model = medsam(sam_model).to(device)
    elif args.method == "treemoeadapter":
        model = MedSegX(sam_model, args.bottleneck_dim, args.embedding_dim, args.expert_num).to(device)
    else:
        raise NotImplementedError("Method {} not implemented!".format(args.method))
    seg_metric = SegmentMetrics(args.metric).to(device)
    
    msx_model = nn.DataParallel(msx_model, device_ids=args.device_ids)
    seg_metric = nn.DataParallel(seg_metric, device_ids=args.device_ids)

    work_dir = join(args.work_dir, args.data_dim, args.task_name)
    epoch = ""
    if args.resume is not None:
        if os.path.isfile(args.resume):
            ## Map model to be loaded to specified single GPU
            print(f"load model from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            msx_model.module.load_parameters(checkpoint["model"])
        work_dir = os.path.dirname(args.resume)
        epoch = "_" + args.resume[:-4].split("_")[-1]
    work_dir = join(work_dir, "ID")
    os.makedirs(work_dir, exist_ok=True)
    
    img_size = msx_model.module.sam.image_encoder.img_size
    img_transform = Resize((img_size, img_size), antialias=True)
    box_transform = ResizeLongestSide(img_size)

    result_total = {}
    meta = {"File": []}
    for m in args.metric:
        result_total[m] = []
        meta[m.upper()] = []
    time_start = time.time()
    
    print(f"save evaluation result to {join(work_dir, 'ID{}.md'.format(epoch))}")
    with open(join(work_dir, 'ID{}.md'.format(epoch)), mode="w") as f:
        f.write("# ID evaluation\n\n")
        data_path = join(args.data_path, args.data_dim, "ID")
        for dataset in sorted(os.listdir(data_path)):
            f.write(f"- {dataset}\n")
            dataset_path = join(data_path, dataset)
            for task in sorted(os.listdir(dataset_path)):
                task_path = join(dataset_path, task)
                
                if 'npy_gts' in os.listdir(task_path):
                    test_dataset = TaskMedSegDB(task_path, train=False)
                    test_dataloader = DataLoader(
                        test_dataset,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=True,
                    )
                    if len(test_dataset) == 0:
                        continue
                
                    # evaluate
                    metric_task = evaluate(msx_model, seg_metric, test_dataloader, 
                                           img_size, img_transform, box_transform, 
                                           dataset, task, result_total=result_total, 
                                           meta=meta, args=args)
                    result_task = ", ".join([f"{k.upper()} ({v:.4f})" for k, v in metric_task.items()])
                    f.write(f"  - {task}: {result_task}\n")
                
                else:
                    f.write(f"  - {task}\n")
                    for sequence in sorted(os.listdir(task_path)):
                        sequence_path = join(task_path, sequence)
                        test_dataset = TaskMedSegDB(sequence_path, train=False)
                        test_dataloader = DataLoader(
                            test_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=True,
                        )
                        if len(test_dataset) == 0:
                            continue
                        
                        # evaluate
                        metric_task = evaluate(msx_model, seg_metric, test_dataloader, 
                                               img_size, img_transform, box_transform, 
                                               dataset, task, sequence, result_total=result_total,
                                               meta=meta, args=args)
                        result_task = ", ".join([f"{k.upper()} ({v:.4f})" for k, v in metric_task.items()])
                        f.write(f"    - {sequence}: {result_task}\n")
                        
        result_total = {k: torch.cat(v).mean().item() for k, v in result_total.items()}
        result_total = ", ".join([f"{k.upper()} ({v:.4f})" for k, v in result_total.items()])
        f.write(f"- ALL\n")
        f.write(f"  - Mean: {result_total}\n")
    
    time_end = time.time()
    print("Time cost: ", (time_end - time_start))
    
    df = pd.DataFrame(meta)
    df.to_csv(f"{work_dir}/ID{epoch}.csv", index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
