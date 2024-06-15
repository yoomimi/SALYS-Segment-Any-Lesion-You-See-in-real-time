import argparse
import os
import time
from pathlib import Path

import lightning as L
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from box import Box
from lightning.fabric.strategies import DDPStrategy

from datetime import datetime
from config import cfg
from dataset import load_datasets
from lightning.fabric.fabric import _FabricOptimizer
from lightning.fabric.loggers import TensorBoardLogger
from losses import DiceLoss
from losses import FocalLoss
from model import MODELS, Model
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from utils import AverageMeter
from utils import calc_iou
import matplotlib.pyplot as plt

torch.set_float32_matmul_precision('high')


def validate(fabric: L.Fabric, model: Model, val_dataloader: DataLoader, epoch: int = 0):
    model.eval()
    ious = AverageMeter()
    f1_scores = AverageMeter()

    with torch.no_grad():
        for iter, data in enumerate(val_dataloader):
            images, prompt_input, gt_masks = data
            num_images = images.size(0)
            pred_masks, _ = model(images, prompt_input)
            for pred_mask, gt_mask in zip(pred_masks, gt_masks):
                pred_mask = pred_mask.sigmoid()
                batch_stats = smp.metrics.get_stats(
                    pred_mask,
                    gt_mask.int(),
                    mode='binary',
                    threshold=0.5,
                )
                batch_iou = smp.metrics.iou_score(*batch_stats, reduction="micro-imagewise")
                batch_f1 = smp.metrics.f1_score(*batch_stats, reduction="micro-imagewise")
                ious.update(batch_iou, num_images)
                f1_scores.update(batch_f1, num_images)
            fabric.print(
                f'Val: [{epoch}] - [{iter}/{len(val_dataloader)}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]'
            )

    fabric.print(f'Validation [{epoch}]: Mean IoU: [{ious.avg:.4f}] -- Mean F1: [{f1_scores.avg:.4f}]')

    fabric.print(f"Saving checkpoint to {cfg.out_dir}")
    state_dict = model.state_dict()
    if fabric.global_rank == 0 and epoch % cfg.save_interval == 0:
        torch.save(state_dict, os.path.join(cfg.out_dir, f"{Path(fabric.logger.log_dir).name}_epoch-{epoch:06d}-f1{f1_scores.avg:.2f}-ckpt.pth"))
    model.train()

    return {"iou_val": ious.avg, "f1_avg": f1_scores.avg}


def train_sam(
        cfg: Box,
        fabric: L.Fabric,
        model: Model,
        optimizer: _FabricOptimizer,
        scheduler: _FabricOptimizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
):
    """The SAM training loop."""

    prompt_type=  cfg.dataset.train.prompt_types[0]
        
    #################################################################################
    #################################################################################
    now = datetime.now()
    formatted_time = now.strftime("%y-%m-%d-%H-%M-%S")
    current_exp_dir = str(cfg.out_dir)+'/'+formatted_time+'_'+prompt_type
    Path(current_exp_dir).mkdir(exist_ok=True, parents=True)
    #################################################################################
    

    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    losses_dice = []
    losses_total = []
    for epoch in range(1, cfg.num_epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        focal_losses = AverageMeter()
        dice_losses = AverageMeter()
        iou_losses = AverageMeter()
        total_losses = AverageMeter()
        end = time.time()
        validated = False

        for iter, data in enumerate(train_dataloader):
            
            ########################################################################
            # if (epoch == 1 or epoch % cfg.eval_interval == 0) and not validated:
            if epoch % cfg.eval_interval == 0 and not validated:
                val_metrics = validate(fabric, model, val_dataloader, epoch)
                fabric.log_dict(val_metrics, step=(epoch - 1) * len(train_dataloader))
                validated = True
            ########################################################################
            torch.cuda.empty_cache()
            
            data_time.update(time.time() - end)
            images, prompt_input, gt_masks = data
            
            batch_size = images.size(0)
            pred_masks, iou_predictions = model(images, prompt_input)
            
            num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
            loss_focal = torch.tensor(0., device=fabric.device)
            loss_dice = torch.tensor(0., device=fabric.device)
            loss_iou = torch.tensor(0., device=fabric.device)
            for pred_mask, gt_mask, iou_prediction in zip(pred_masks, gt_masks, iou_predictions):
                batch_iou = calc_iou(pred_mask, gt_mask)
                loss_focal += focal_loss(pred_mask, gt_mask) / num_masks
                loss_dice += dice_loss(pred_mask, gt_mask) / num_masks
                loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction='sum') / num_masks

            loss_total = 20. * loss_focal + loss_dice + loss_iou
            optimizer.zero_grad()
            fabric.backward(loss_total)
            optimizer.step()
            scheduler.step()
            batch_time.update(time.time() - end)
            end = time.time()

            focal_losses.update(loss_focal.item(), batch_size)
            dice_losses.update(loss_dice.item(), batch_size)
            iou_losses.update(loss_iou.item(), batch_size)
            total_losses.update(loss_total.item(), batch_size)


            if iter % 25 == 0 : 
            # if iter % 1 == 0 : 
                losses_dice, losses_total = log_progress(prompt_type, images, prompt_input, pred_masks, gt_masks, loss_dice.item(),loss_total.item(), losses_dice, losses_total, epoch, iter,current_exp_dir)
                
            fabric.print(f'Epoch: [{epoch}][{iter + 1}/{len(train_dataloader)}]'
                         f' | Time [{batch_time.val:.3f}s ({batch_time.moving_avg:.3f}s)]'
                         f' | Data [{data_time.val:.3f}s ({data_time.moving_avg:.3f}s)]'
                         f' | Focal Loss [{focal_losses.val:.4f} ({focal_losses.moving_avg:.4f})]'
                         f' | Dice Loss [{dice_losses.val:.4f} ({dice_losses.moving_avg:.4f})]'
                         f' | IoU Loss [{iou_losses.val:.4f} ({iou_losses.moving_avg:.4f})]'
                         f' | Total Loss [{total_losses.val:.4f} ({total_losses.moving_avg:.4f})]')
            fabric.log_dict({"focal_loss": focal_losses.val,
                             "dice_loss": dice_losses.val,
                             "iou_loss": iou_losses.val,
                             "total_loss": total_losses.val,
                             "batch_time": batch_time.val,
                             "data_time": data_time.val,
                             }, step=(epoch - 1) * len(train_dataloader) + iter)
        fabric.log_dict({"lr": scheduler.get_last_lr()[0]}, step=epoch * len(train_dataloader))
            
        
        

def log_progress(prompt_type, img_orig, prompts, pred_masks, gt_masks, loss1,loss2, loss_list1, loss_list2, epoch, iteration, current_exp_dir):
    
    img_size = img_orig.shape[-2:]

    loss_list1.append(loss1)
    loss_list2.append(loss2)
    
    
    img_orig = img_orig[0].cpu().numpy()
    prompts = prompts[0]

    pred_masks = pred_masks[0][0].detach().cpu().numpy()
    pred_masks =  1 / (1 + np.exp(-pred_masks)) ## do sigmoid

    gt_masks = gt_masks[0][0].cpu().numpy()
    
    prompt_img = np.zeros_like(img_orig)
    
    if prompt_type == 'boxes':
        y1, x1, y2, x2  = prompts["boxes"].detach().cpu().numpy()[0]
        y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)
        prompt_img = np.copy(img_orig)
        prompt_img[0, y1:y2, x1] = 1.0  
        prompt_img[0, y1:y2, x1-1] = 1.0  
        prompt_img[0, y1:y2, x1+1] = 1.0  
        
        prompt_img[0, y1:y2, x2] = 1.0  
        prompt_img[0, y1:y2, x2-1] = 1.0
        prompt_img[0, y1:y2, x2-1] = 1.0
        
        
        prompt_img[0, y1, x1:x2] = 1.0  
        prompt_img[0, y1-1, x1:x2] = 1.0
        prompt_img[0, y1+1, x1:x2] = 1.0
        
        prompt_img[0, y2, x1:x2] = 1.0 
        prompt_img[0, y2-1, x1:x2] = 1.0 
        prompt_img[0, y2+1, x1:x2] = 1.0 
        
        prompt_img = prompt_img[None, ...]
        
    elif prompt_type == 'masks':
        prompt_img = prompts["masks"].detach().cpu().numpy()
        
        prompt_img=  prompt_img[0,0][None,None, ...]
        


    overlay_pred = img_orig.copy()
    overlay_pred[1] += np.clip(pred_masks,0, 1) 
    overlay_gt = img_orig.copy()
    overlay_gt[1] += np.clip(gt_masks ,0, 1) 
    
    
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    img_orig = np.transpose(img_orig, (1, 2, 0))
    overlay_pred = np.transpose(overlay_pred, (1, 2, 0))
    overlay_gt = np.transpose(overlay_gt, (1, 2, 0))
    
    
    if prompt_type == 'boxes' or prompt_type == 'masks':
        prompt_img_rescaled = (prompt_img - np.min(prompt_img)) / (np.max(prompt_img) - np.min(prompt_img))
        prompt_img_rescaled = F.interpolate(torch.tensor(prompt_img_rescaled), (img_orig.shape[0],img_orig.shape[1]), mode='bilinear', align_corners=False).squeeze(0).detach().cpu().numpy()
        prompt_img_rescaled = np.transpose(prompt_img_rescaled, (1, 2, 0))

    ax = axes[0, 0]; ax.imshow(img_orig);    ax.set_title(f"Input Image")
    
    if prompt_type =='gaze_points' or prompt_type == 'points':
        ax = axes[0, 1];
        ax.imshow(img_orig);
        xs = []
        ys = []
        for pt in prompts["points"][0][0]:
            
            x = int(pt[0].detach().cpu().numpy())
            y = int(pt[1].detach().cpu().numpy())
            xs.append(x)
            ys.append(y)

        ax.scatter(xs, ys, c='r', s=10);  ax.set_title(f"Prompt Image")
        
        
    elif prompt_type =='masks' : 
        ax = axes[0, 1]; 
        ax.imshow(img_orig);  
        ax.imshow(prompt_img_rescaled, alpha=0.5, cmap='jet');  
        
        ax.set_title(f"Prompt Image")    
        
    else : 
        ax = axes[0, 1]; ax.imshow(prompt_img_rescaled);  ax.set_title(f"Prompt Image")    
    
    
    ax = axes[0, 2]; ax.imshow(pred_masks);  ax.set_title(f"Prediction")
    ax = axes[0, 3]; ax.imshow(gt_masks);    ax.set_title(f"GT")
    ax = axes[1, 0]; ax.plot(loss_list1);    ax.set_title(f"loss_Dice")
    ax = axes[1, 1]; ax.plot(loss_list2);    ax.set_title(f"loss_Total")
    ax = axes[1, 2]; ax.imshow(overlay_pred);  ax.set_title(f"IMG + Prediction")
    ax = axes[1, 3]; ax.imshow(overlay_gt);    ax.set_title(f"IMG + GT")
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
    
    plt.tight_layout()
    fig_save_path = current_exp_dir + '/epoch_' + str(epoch) +'_iter_'+ str(iteration)+'.png'
    plt.savefig(fig_save_path) 
    plt.show()  
    plt.close(fig)
    return loss_list1, loss_list2

        


def configure_opt(cfg: Box, model, num_steps_per_epoch):
    def lr_lambda(step):
        if step < cfg.opt.warmup_steps:
            return step / cfg.opt.warmup_steps
        elif step < cfg.opt.steps[0]:
            return 1.0
        elif step < cfg.opt.steps[1]:
            return 1 / cfg.opt.decay_factor
        else:
            return 1 / (cfg.opt.decay_factor ** 2)

    def lr_lambda_exp(step):
        if step < cfg.opt.warmup_steps:
            return step / cfg.opt.warmup_steps
        else:
            return 0.95 ** (step // num_steps_per_epoch)

    optimizer = torch.optim.AdamW(model.get_parameters(), lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_exp)

    return optimizer, scheduler


def main(cfg: Box) -> None:
    
    
    Path(cfg.out_dir).mkdir(exist_ok=True, parents=True)
    
    
    
    fabric = L.Fabric(accelerator="auto",
                      devices=cfg.num_devices,
                      strategy=DDPStrategy(start_method="popen", find_unused_parameters=True),
                      loggers=[TensorBoardLogger(cfg.out_dir, name=cfg.config_name)])
    cfg.out_dir = Path(cfg.out_dir, cfg.config_name)
    cfg.out_dir.mkdir(exist_ok=True, parents=True)
    fabric.launch()
    fabric.seed_everything((np.random.randint(1, 420) if args.seed else 1337) + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(cfg.out_dir, exist_ok=True)

    with fabric.device:
        print(cfg.model.name,'cfg.model.name')
        model = MODELS[cfg.model.name](cfg)
        model.setup()
    
    
    train_data, val_data = load_datasets(cfg, model.get_img_size())
    
    train_data = fabric._setup_dataloader(train_data)
    val_data = fabric._setup_dataloader(val_data)

    optimizer, scheduler = configure_opt(cfg, model, num_steps_per_epoch=len(train_data))
    model, optimizer = fabric.setup(model, optimizer)

    train_sam(cfg, fabric, model, optimizer, scheduler, train_data, val_data)
    val_metrics = validate(fabric, model, val_data, epoch=cfg.num_epochs)
    fabric.log_dict(val_metrics, step=(cfg.num_epochs - 1) * len(train_data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Finetune SAM using the corresponding training config")
    parser.add_argument("--config", default="configs/base_config_chest.yaml", type=str,
                        help="Path to .yaml file containing the config.")
    parser.add_argument("--seed", action="store_true", help="if set, use random seed for init")

    args = parser.parse_args()
    if args.config is not None:
        try:
            cfg = Box.from_yaml(filename=str(Path(args.config).absolute()))
        except Exception as e:
            print("Failed to load config:")
            print(e)
            print("Using default config instead")
        cfg["config_name"] = Path(args.config).stem
    else:
        cfg["config_name"] = "internal_config"
    if "num_nodes" not in cfg.keys():
        cfg["num_nodes"] = 1
    main(cfg)
