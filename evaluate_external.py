# -*- coding: utf-8 -*-
"""
MedSedX external evaluating script
"""

# setup environment
import argparse
import os
join = os.path.join
import time

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
import pandas as pd
from tqdm import tqdm

from segment_anything import sam_model_registry, sam_model_checkpoint
from segment_anything.utils.transforms import ResizeLongestSide
from model import *
from data.dataset import TaskMedSegDB
from utils.metric import SegmentMetrics

# setup parser
parser = argparse.ArgumentParser("MedSedX external evaluating", add_help=False)
# model
parser.add_argument("--checkpoint", type=str, default="./playground/SAM",
                    help="path to SAM checkpoint folder")
parser.add_argument("--model_type", type=str, default="vit_b",
                    help="SAM model scale (e.g vit_b, vit_l, vit_h)")
parser.add_argument("--model_weight", type=str, default="./playground/MedSegX/medsegx_vit_b.pth",
                    help="path to MedSegX model weight")
parser.add_argument("--method", type=str, default="medsegx")
parser.add_argument("--bottleneck_dim", type=int, default=16)
parser.add_argument("--embedding_dim", type=int, default=16)
parser.add_argument("--expert_num", type=int, default=4)
# data
parser.add_argument("--data_path", type=str, default="./playground/MedSegDB/example",
                    help="path to MedSegDB data folder")
parser.add_argument("--shift_type", type=str, default="cross_site",
                    help="external shift type (e.g cross_site, cross_task)")
parser.add_argument("--metric", type=str, default=["dsc", "hd"], nargs='+',
                    help="evaluation metrics (e.g dsc, hd)")
# eval
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--device_ids", type=int, default=[0,1,2,3,4,5,6,7], nargs='+',
                    help="device ids assignment (e.g 0 1 2 3)")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=32)


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
            
            # handle ambiguous segmentation
            for idx in range(model.module.sam.mask_decoder.num_multimask_outputs):
                result_batch = metric(mask[:, idx].unsqueeze(1), label)
                for m in args.metric:
                    result_list[m].append(result_batch[m])
            
            dsc, max_idx = torch.stack(result_list["dsc"], dim=0).max(dim=0)
            for m in args.metric:
                if m == "dsc":
                    result = dsc
                else:
                    # select other metrics based on the best DSC
                    result = torch.stack(result_list[m], dim=0)
                    result = result[max_idx, torch.arange(result.shape[1])]
                metric_dict[m] = result.mean().item()
                metric_list[m].append(result)
                result_task[m].append(result)
                result_total[m].append(result)
            
            pbar.set_postfix(metric_dict)
            metric_list = {m: torch.cat(v) for m, v in metric_list.items()}
            for idx, name in enumerate(data["name"]):
                file = name.replace(f"{args.data_path}/external/{args.shift_type}/", "").replace("inference/", "").replace("npy_gts/", "")
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
        model = MedSAM(sam_model).to(device)
    elif args.method == "medsegx":
        model = MedSegX(sam_model, args.bottleneck_dim, args.embedding_dim, args.expert_num).to(device)
    else:
        raise NotImplementedError("Method {} not implemented!".format(args.method))
    seg_metric = SegmentMetrics(args.metric).to(device)
    
    model = nn.DataParallel(model, device_ids=args.device_ids)
    seg_metric = nn.DataParallel(seg_metric, device_ids=args.device_ids)

    if os.path.isfile(args.model_weight):
        ## Map model to be loaded to specified single GPU
        print(f"load model from {args.model_weight}")
        checkpoint = torch.load(args.model_weight, map_location=device)
        model.module.load_parameters(checkpoint["model"])
    else:
        raise FileNotFoundError(f"model weight {args.model_weight} not found!")
    work_dir = os.path.dirname(args.model_weight)
    work_dir = join(work_dir, "example", "external", args.shift_type)
    os.makedirs(work_dir, exist_ok=True)
    
    img_size = model.module.sam.image_encoder.img_size
    img_transform = Resize((img_size, img_size), antialias=True)
    box_transform = ResizeLongestSide(img_size)
    
    # cross-site (task, dataset) list
    site_list = [
        ("CBCT_Tooth", "DentalPanoramicRadiographsDataset"), ("CT_GallBladder", "Totalsegmentator_dataset"), 
        ("CT_LeftKidney", "BTCV"), ("CT_RightKidney", "BTCV"), 
        ("CT_LeftLung", "COVID_19_Seg"), ("CT_RightLung", "COVID_19_Seg"), 
        ("CT_Liver", "SLIVER07"), ("CT_Pancreas", "FLARE22Train"), 
        ("CT_Stomach", "FLARE22Train"), ("Fundus_OpticCup", "inFundus"), 
        ("Fundus_OpticDisc", "ADAM"), ("MRI_LeftAtrium", "HeartSegMRI"), 
        ("MRI_Prostate", "prostate158"), ("MRI_RightVentricle", "MM"), 
        ("MRI_RightVentricularMyocardium", "MM"), ("MRI_Spleen", "CHAOS"), 
        ("xray_LeftLung", "ShenZhen"), ("xray_RightLung", "ShenZhen"), 
    ]
    # cross-task (task, dataset) list
    task_list = [
        ("CT_ColonCancer", "MSD_Task10"), ("CT_KidneyTumor", "KiPA22"), 
        ("CT_KidneyTumor", "KiTS19"), ("CT_LiverCancer", "MSD_Task03"), 
        ("CT_LiverCancer", "MSD_Task08"), ("CT_LungCancer", "LNDb"), 
        ("CT_PancreasCancer", "MSD_Task07"), ("MRI_ProstateTumor", "prostate158"), 
        ("MRI_ProstateTumor", "PicaiBaseline"), ("MRI_VestibularSchwannoma", "CrossMoDA2022"),
    ]
    if args.shift_type == "cross_site":
        check_list = site_list
    elif args.shift_type == "cross_task":
        check_list = task_list

    result_total = {}
    meta = {"File": []}
    for m in args.metric:
        result_total[m] = []
        meta[m.upper()] = []
    time_start = time.time()
    
    print(f"save evaluation result to {join(work_dir, '{}.md'.format(args.shift_type))}")
    with open(join(work_dir, '{}.md'.format(args.shift_type)), mode="w") as f:
        f.write(f"# external {args.shift_type} evaluation\n\n")
        data_path = join(args.data_path, "external", args.shift_type)
        # iterate over tasks
        for task in sorted(os.listdir(data_path)):
            if task not in [item[0] for item in check_list]:
                continue
            f.write(f"- {task}\n")
            task_path = join(data_path, task)
            # iterate over datasets
            for dataset in sorted(os.listdir(task_path)):
                dataset_path = join(task_path, dataset)
                if (task, dataset) not in check_list:
                    continue
                
                # do not have different sequences
                if 'inference' in os.listdir(dataset_path):
                    test_dataset = TaskMedSegDB(join(dataset_path, "inference"), train=False)
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
                    metric_task = evaluate(model, seg_metric, test_dataloader, 
                                           img_size, img_transform, box_transform, 
                                           dataset, task, result_total=result_total, 
                                           meta=meta, args=args)
                    result_task = ", ".join([f"{k.upper()} ({v:.4f})" for k, v in metric_task.items()])
                    f.write(f"  - {dataset}: {result_task}\n")
                
                # have different sequences
                else:
                    f.write(f"  - {dataset}\n")
                    for sequence in sorted(os.listdir(dataset_path)):
                        if args.shift_type == "cross_site":
                            if dataset == "CHAOS" and sequence != "T2W":
                                continue
                        elif args.shift_type == "cross_task":
                            if dataset == "prostate158" and sequence != "T2":
                                continue
                            elif dataset == "PicaiBaseline" and sequence != "T2W":
                                continue
                        sequence_path = join(dataset_path, sequence)
                        test_dataset = TaskMedSegDB(join(sequence_path, "inference"), train=False)
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
                        metric_task = evaluate(model, seg_metric, test_dataloader, 
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
    print(f"Time cost: {time_end - time_start:.0f} s")
    
    # record instance-level results
    df = pd.DataFrame(meta)
    df.to_csv(f"{work_dir}/{args.shift_type}.csv", index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
