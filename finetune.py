# -*- coding: utf-8 -*-
"""
MedSedX fine-tuning script
"""

# setup environment
import argparse
import copy
from datetime import datetime
import os
join = os.path.join
import random
import json

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Resize
import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm

from segment_anything import sam_model_registry, sam_model_checkpoint
from segment_anything.utils.transforms import ResizeLongestSide
from model import *
from data.dataset import TaskMedSegDB
from utils.loss import DiceBCELoss
from utils.logger import get_logger
from utils.metric import SegmentMetrics
from utils.scheduler import adjust_learning_rate

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
parser = argparse.ArgumentParser("MedSedX finetuning", add_help=False)
parser.add_argument("--model_type", type=str, default="vit_b")
parser.add_argument("--task_name", type=str, default="MedSegX_256")
parser.add_argument("--method", type=str, default="medsegx")
parser.add_argument("--bottleneck_dim", type=int, default=16)
parser.add_argument("--embedding_dim", type=int, default=16)
parser.add_argument("--expert_num", type=int, default=4)
parser.add_argument("--checkpoint", type=str, default="/path/to/SAM")
parser.add_argument("--data_path", type=str, default="/path/to/MedSegDB")
parser.add_argument("--shift_type", type=str, default="cross_site")
parser.add_argument("--data_dim", type=str, default="2D")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--device_ids", type=int, default=[0,1,2,3,4,5,6,7], nargs='+',
                    help="device ids assignment (e.g 0 1 2 3)")
parser.add_argument("--work_dir", type=str, default="./work_dir")
# train
parser.add_argument("--num_epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_workers", type=int, default=32)
parser.add_argument("--validation", type=str, default="val")
parser.add_argument("--resume", type=str, default=None, 
                    help="Resuming training from checkpoint")
# Optimizer parameters
parser.add_argument("--weight_decay", type=float, default=0.01, 
                    help="weight decay (default: 0.01)")
parser.add_argument("--lr", type=float, default=0.00005, metavar="LR", 
                    help="learning rate (absolute lr default: 0.00005)")
parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0')
parser.add_argument("--warmup_epochs", type=int, default=5, 
                    help="number of warmup epochs (default: 5)")
parser.add_argument("--lr_scheduler", type=str, default="cosine", 
                    help="learning rate scheduler type (default: cosine)")
parser.add_argument("--use_amp", action="store_true", default=False, 
                    help="use amp")


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
    dsc_metric = SegmentMetrics(["dsc"]).to(device)
    
    model = nn.DataParallel(model, device_ids=args.device_ids)
    dsc_metric = nn.DataParallel(dsc_metric, device_ids=args.device_ids)
    
    work_dir = join(args.work_dir, args.data_dim, args.task_name)
    if args.resume is not None:
        if os.path.isfile(args.resume):
            ## Map model to be loaded to specified single GPU
            print(f"load model from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.module.load_parameters(checkpoint["model"])
        work_dir = os.path.dirname(args.resume)
    work_dir = join(work_dir, f"finetune_{args.lr}", args.shift_type)
    os.makedirs(work_dir, exist_ok=True)
    logger = get_logger(log_file=os.path.join(work_dir, 'output.log'))
    logger.info(f"args: {json.dumps(vars(args), indent=2)}")

    logger.info("Model: %s" % str(model))
    logger.info(
        "Number of total parameters: %d" % (
            sum(p.numel() for p in model.parameters()))
    )  # 93735472: (image_encoder: 89670912, prompt_encoder: 6220, mask_decoder: 4058340)
    logger.info(
        "Number of trainable parameters: %d" % (
            sum(p.numel() for p in model.parameters() if p.requires_grad))
    )
    
    img_size = model.module.sam.image_encoder.img_size
    img_transform = Resize((img_size, img_size), antialias=True)
    box_transform = ResizeLongestSide(img_size)
    
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
    
    data_path = join(args.data_path, args.data_dim, "OOD", args.shift_type)
    for task in sorted(os.listdir(data_path)):
        task_path = join(data_path, task)
        for dataset in sorted(os.listdir(task_path)):
            dataset_path = join(task_path, dataset)
            if (task, dataset) not in check_list:
                continue
            logger.info(f"Task: {task}, Dataset: {dataset}")
            
            if "inference" not in os.listdir(dataset_path):
                if args.shift_type == "cross_site":
                    dataset_path = join(dataset_path, "T2W")
                elif args.shift_type == "cross_task":
                    if "prostate158" in dataset_path:
                        dataset_path = join(dataset_path, "T2")
                    elif "PicaiBaseline" in dataset_path:
                        dataset_path = join(dataset_path, "T2W")
            
            test_dataset = TaskMedSegDB(join(dataset_path, "inference"), train=False)
            
            finetune_path = join(dataset_path, "finetune")
            for percent in sorted(os.listdir(finetune_path)):
                if percent != "005_percent":
                    continue
                train_dataset = TaskMedSegDB(join(finetune_path, percent), train=True)
                if len(train_dataset) == 0:
                    continue
                
                if args.validation == "val":
                    train_size = int(len(train_dataset) * 0.8)
                    val_size = len(train_dataset) - train_size
                    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
                    val_dataset.dataset.train = False
                else:
                    val_dataset = test_dataset
                
                logger.info(f"Number of {percent} training samples: {len(train_dataset)}")
                train_dataloader = DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=True,
                )
                logger.info(f"Number of {percent} validation samples: {len(val_dataset)}")
                val_dataloader = DataLoader(
                    val_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=True,
                )
                
                finetune_model = copy.deepcopy(model)
                optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, finetune_model.parameters()),
                    lr=args.lr, weight_decay=args.weight_decay
                )
                criterion = DiceBCELoss(sigmoid=True, squared_pred=True, reduction='none')

                num_epochs = args.num_epochs
                best_dsc = 0
                best_epoch = -1
                loss_log = []
                lr_log = []
                dsc_log = []
                
                save_dir = join(work_dir, task, dataset, percent)
                if "T2W" in dataset_path:
                    save_dir = join(work_dir, task, dataset, "T2W", percent)
                elif "T2" in dataset_path:
                    save_dir = join(work_dir, task, dataset, "T2", percent)
                os.makedirs(save_dir, exist_ok=True)
                if args.use_amp:
                    scaler = torch.cuda.amp.GradScaler()
                
                for epoch in range(num_epochs):
                    # train
                    epoch_loss = 0
                    step = 0
                    finetune_model.train()
                    pbar_train = tqdm(train_dataloader)
                    pbar_train.set_description(f"Epoch [{epoch}/{num_epochs}] Tune")
                    for data, label in pbar_train:
                        optimizer.zero_grad()
                        step += 1
                        adjust_learning_rate(optimizer, epoch + step / len(train_dataloader), args)
            
                        if data["img"].shape[-1] != img_size:
                            data["box"] = box_transform.apply_boxes_torch((data["box"].reshape(-1, 2, 2)), 
                                                                            data["img"].shape[-2:]).reshape(-1, 4)
                            data["img"] = img_transform(data["img"])
                        data["img"] = data["img"].to(device, non_blocking=True)
                        data["box"] = data["box"].to(device, non_blocking=True)
                        label = label.to(device, non_blocking=True)
                        
                        if args.use_amp:
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                mask_pred = finetune_model(data)
                        else:
                            mask_pred = finetune_model(data)
                        if mask_pred.shape[-1] != label.shape[-1]:
                            mask_pred = F.interpolate(mask_pred, size=label.shape[-1], mode="bilinear", antialias=True)
                        
                        losses = []
                        for i in range(finetune_model.module.sam.mask_decoder.num_multimask_outputs):
                            output = mask_pred[:, i].unsqueeze(1)
                            loss = criterion(output.float(), label)
                            losses.append(loss)
                        loss = torch.stack(losses, dim=0).min(dim=0)[0]
                        loss = loss.mean()
                        if args.use_amp:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            optimizer.step()

                        epoch_loss += loss.item()
                        lr = optimizer.state_dict()['param_groups'][0]['lr']
                        pbar_train.set_postfix({"lr": lr, "loss": loss.item()})

                    lr_log.append(lr)
                    epoch_loss /= step
                    loss_log.append(epoch_loss)
                    print(
                        f'Time: {datetime.now().strftime("%Y/%m/%d-%H:%M")}, Epoch: {epoch}, Loss: {epoch_loss}'
                    )
        
                    # validation
                    epoch_dsc = 0
                    size = 0
                    finetune_model.eval()
                    pbar_val = tqdm(val_dataloader)
                    pbar_val.set_description(f"Epoch [{epoch}/{num_epochs}] Val")
                    with torch.no_grad():
                        for data, label in pbar_val:
                            if data["img"].shape[-1] != img_size:
                                data["box"] = box_transform.apply_boxes_torch((data["box"].reshape(-1, 2, 2)), 
                                                                                data["img"].shape[-2:]).reshape(-1, 4)
                                data["img"] = img_transform(data["img"])
                            data["img"] = data["img"].to(device, non_blocking=True)
                            data["box"] = data["box"].to(device, non_blocking=True)
                            label = label.to(device, non_blocking=True)
                            
                            mask_pred = finetune_model(data)
                            if mask_pred.shape[-1] != label.shape[-1]:
                                mask_pred = F.interpolate(mask_pred, size=label.shape[-1], mode="bilinear", antialias=True)
                            mask_prob = torch.sigmoid(mask_pred)
                            mask = (mask_prob > 0.5).bool()
                            
                            dsc_ambiguous = []
                            for idx in range(model.module.sam.mask_decoder.num_multimask_outputs):
                                dsc_ambiguous.append(dsc_metric(mask[:, idx].unsqueeze(1), label)["dsc"])
                            dsc = torch.stack(dsc_ambiguous, dim=0).max(dim=0)[0]
                            
                            epoch_dsc += dsc.sum().item()
                            size += dsc.shape[0]
                            pbar_val.set_postfix({"dsc": dsc.mean().item()})
                    
                    epoch_dsc /= size
                    dsc_log.append(epoch_dsc)
                    print(
                        f'Time: {datetime.now().strftime("%Y/%m/%d-%H:%M")}, Epoch: {epoch}, DSC: {epoch_dsc}'
                    )
                    
                    logger.info(f"Epoch [{epoch}] - LR: {lr}, Loss: {epoch_loss}, DSC: {epoch_dsc}")
                    
                    if epoch_dsc > best_dsc:
                        best_dsc = epoch_dsc
                        best_epoch = epoch
                        checkpoint = {
                            "model": finetune_model.module.save_parameters(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                        }
                        torch.save(checkpoint, join(save_dir, "model_best.pth"))
        
                    # plot loss
                    plt.plot(loss_log)
                    plt.title("Training Loss")
                    plt.xlabel("Epoch")
                    plt.ylabel("Loss")
                    plt.savefig(join(save_dir, "train_loss.png"))
                    plt.close()
                    
                    # plot lr
                    plt.plot(lr_log)
                    plt.title("Learning Rate")
                    plt.xlabel("Epoch")
                    plt.ylabel("LR")
                    plt.savefig(join(save_dir, "lr.png"))
                    plt.close()
                    
                    # plot dsc
                    plt.plot(dsc_log)
                    plt.title("Validation DSC")
                    plt.xlabel("Epoch")
                    plt.ylabel("DSC")
                    plt.savefig(join(save_dir, "val_dsc.png"))
                    plt.close()
                    
                    with open(join(save_dir, "result.log"), mode='w') as f:
                        for e in range(len(loss_log)):
                            f.write(f'Epoch [{e}] - LR: {lr_log[e]}, Loss: {loss_log[e]}, DSC: {dsc_log[e]}\n')
                        f.write(f'Best epoch: {best_epoch}, Best DSC: {best_dsc}\n')
                logger.info(f"Best epoch: {best_epoch}, Best DSC: {best_dsc}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
