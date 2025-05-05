# -*- coding: utf-8 -*-
"""
MedSedX pre-training script
"""

# setup environment
import argparse
from datetime import datetime, timedelta
import os
join = os.path.join
import random
import time
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
from data.dataset import GeneralMedSegDB
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
parser = argparse.ArgumentParser("MedSedX training", add_help=False)
parser.add_argument("--model_type", type=str, default="vit_b")
parser.add_argument("--task_name", type=str, default="MedSegX_256")
parser.add_argument("--method", type=str, default="medsegx")
parser.add_argument("--bottleneck_dim", type=int, default=16)
parser.add_argument("--embedding_dim", type=int, default=16)
parser.add_argument("--expert_num", type=int, default=4)
parser.add_argument("--checkpoint", type=str, default="/path/to/SAM")
parser.add_argument("--data_path", type=str, default="/path/to/MedSegDB")
parser.add_argument("--data_dim", type=str, default="2D")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--device_ids", type=int, default=[0,1,2,3,4,5,6,7], nargs='+',
                    help="device ids assignment (e.g 0 1 2 3)")
parser.add_argument("--work_dir", type=str, default="./work_dir")
# train
parser.add_argument("--num_epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--num_workers", type=int, default=64)
parser.add_argument("--resume", type=str, default=None, 
                    help="Resuming training from checkpoint")
# Optimizer parameters
parser.add_argument("--weight_decay", type=float, default=0., 
                    help="weight decay (default: 0.)")
parser.add_argument("--lr", type=float, default=0.001, metavar="LR", 
                    help="learning rate (absolute lr default: 0.001)")
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
    os.makedirs(work_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=work_dir)
    logger = get_logger(log_file=os.path.join(work_dir, 'output.log'))
    logger.info(f"args: {json.dumps(vars(args), indent=2)}")

    logger.info("Model: %s" % str(model))
    logger.info(
        "Number of total parameters: %d" % (
            sum(p.numel() for p in model.parameters()))
    )
    logger.info(
        "Number of trainable parameters: %d" % (
            sum(p.numel() for p in model.parameters() if p.requires_grad))
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    pos_weight = None
    # pos_weight = torch.tensor([81018774276/3103300860], dtype=torch.float32, device=device)
    criterion = DiceBCELoss(sigmoid=True, squared_pred=True, reduction='none', pos_weight=pos_weight)
    logger.info("Criterion: %s" % str(criterion))

    train_dataset = GeneralMedSegDB(join(args.data_path, args.data_dim, "pretrain"), train=True)
    logger.info(f"Number of training samples: {len(train_dataset)}")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_dataset = GeneralMedSegDB(join(args.data_path, args.data_dim, "ID"), train=False)
    logger.info(f"Number of validation samples: {len(val_dataset)}")
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    img_size = model.module.sam.image_encoder.img_size
    img_transform = Resize((img_size, img_size), antialias=True)
    box_transform = ResizeLongestSide(img_size)

    num_epochs = args.num_epochs
    start_epoch = 0
    best_loss = 1e10
    best_dsc = 0
    best_epoch = -1
    loss_log = []
    lr_log = []
    dsc_log = []
    
    if args.resume is not None:
        if os.path.isfile(args.resume):
            ## Map model to be loaded to specified single GPU
            print(f"load model from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            model.module.load_parameters(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    
    start_time = time.time()
    for epoch in range(start_epoch, num_epochs):
        # train
        epoch_loss = 0
        step = 0
        model.train()
        pbar_train = tqdm(train_dataloader)
        pbar_train.set_description(f"Epoch [{epoch}/{num_epochs}] Train")
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
                    mask_pred = model(data)
            else:
                mask_pred = model(data)
            if mask_pred.shape[-1] != label.shape[-1]:
                mask_pred = F.interpolate(mask_pred, size=label.shape[-1], mode="bilinear", antialias=True)
            
            losses = []
            for i in range(model.module.sam.mask_decoder.num_multimask_outputs):
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
            
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((epoch + step / len(train_dataloader)) * 1000)
            log_writer.add_scalar('batch/lr', lr, epoch_1000x)
            log_writer.add_scalar('batch/loss', loss.item(), epoch_1000x)
        
        lr_log.append(lr)
        epoch_loss /= step
        loss_log.append(epoch_loss)
        log_writer.add_scalar('epoch/lr', lr, epoch + 1)
        log_writer.add_scalar('epoch/loss', epoch_loss, epoch + 1)
        print(
            f'Time: {datetime.now().strftime("%Y/%m/%d-%H:%M")}, Epoch: {epoch}, Loss: {epoch_loss}'
        )
        
        ## save the latest model
        checkpoint = {
            "model": model.module.save_parameters(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, join(work_dir, "model_latest.pth"))
        ## save the lowest model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint = {
                "model": model.module.save_parameters(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(work_dir, "model_lowest.pth"))
        ## save the model
        if True:
            checkpoint = {
                "model": model.module.save_parameters(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(work_dir, f"model_{epoch}.pth"))
        
        # validation
        epoch_dsc = 0
        size = 0
        model.eval()
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
                
                mask_pred = model(data)
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
        log_writer.add_scalar('epoch/dsc', epoch_dsc, epoch + 1)
        print(
            f'Time: {datetime.now().strftime("%Y/%m/%d-%H:%M")}, Epoch: {epoch}, DSC: {epoch_dsc}'
        )
        
        ## save the best model
        if epoch_dsc > best_dsc:
            best_dsc = epoch_dsc
            best_epoch = epoch
            checkpoint = {
                "model": model.module.save_parameters(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(work_dir, "model_best.pth"))
        
        # plot loss
        plt.plot(loss_log)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(join(work_dir, "train_loss.png"))
        plt.close()
        
        # plot lr
        plt.plot(lr_log)
        plt.title("Learning Rate")
        plt.xlabel("Epoch")
        plt.ylabel("LR")
        plt.savefig(join(work_dir, "lr.png"))
        plt.close()
        
        # plot dsc
        plt.plot(dsc_log)
        plt.title("Validation DSC")
        plt.xlabel("Epoch")
        plt.ylabel("DSC")
        plt.savefig(join(work_dir, "val_dsc.png"))
        plt.close()
        
        logger.info(f"Epoch [{epoch}] - LR: {lr}, Loss: {epoch_loss}, DSC: {epoch_dsc}")
        log_writer.flush()
    
    total_time = time.time() - start_time
    total_time_str = str(timedelta(seconds=int(total_time)))
    logger.info(f"Best epoch: {best_epoch}, Best DSC: {best_dsc}")
    logger.info(f"Time cost: {total_time_str}")
    
    # os.system(f"python evaluate_id.py --method {args.method} --bottleneck_dim {args.bottleneck_dim} --embedding_dim {args.embedding_dim} --resume {join(work_dir, 'model_best.pth')}")
    # os.system(f"python evaluate_ood.py --method {args.method} --bottleneck_dim {args.bottleneck_dim} --embedding_dim {args.embedding_dim} --shift_type distribution_shift --resume {join(work_dir, 'model_best.pth')}")
    # os.system(f"python evaluate_ood.py --method {args.method} --bottleneck_dim {args.bottleneck_dim} --embedding_dim {args.embedding_dim} --shift_type task_shift --resume {join(work_dir, 'model_best.pth')}")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
