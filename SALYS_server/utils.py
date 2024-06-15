import os

import cv2
import torch
from box import Box
from model import Model
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks
from tqdm import tqdm
from collections import deque

class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, window=10):
        self.reset()
        self.window_size = window
        self.mavg_buffer = deque(maxlen=self.window_size)

    def reset(self):
        self.val = 0
        self.avg = 0
        self.moving_avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.mavg_buffer.append(val)
        self.moving_avg = sum(self.mavg_buffer) / self.window_size
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor):
    pred_mask = (pred_mask >= 0.5).float()
    intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=(1, 2))
    union = torch.sum(pred_mask, dim=(1, 2)) + torch.sum(gt_mask, dim=(1, 2)) - intersection
    epsilon = 1e-7
    batch_iou = intersection / (union + epsilon)

    batch_iou = batch_iou.unsqueeze(1)
    return batch_iou


def draw_image(image, masks, boxes, labels, alpha=0.4):
    image = torch.from_numpy(image).permute(2, 0, 1)
    if boxes is not None:
        image = draw_bounding_boxes(image, boxes, colors=['red'] * len(boxes), labels=labels, width=2)
    if masks is not None:
        image = draw_segmentation_masks(image, masks=masks, colors=['red'] * len(masks), alpha=alpha)
    return image.numpy().transpose(1, 2, 0)

