import torch 
from torch import Tensor
import numpy as np 
from typing import Literal
import torch.nn.functional as F
# losses
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dot_prod(x: Tensor, y: Tensor, reduce: bool=True) -> Tensor:
    dot_prod = torch.sum(x * y, dim=-1)
    if reduce:
        return torch.mean(dot_prod)
    return dot_prod

def dir_dot_loss(x: Tensor, y: Tensor, reduce: bool=True) -> Tensor:
    return 1 - dot_prod(x, y, reduce=reduce)

def _img2mse(x: Tensor, y: Tensor, reduce: bool=True) -> Tensor:
    l = (x - y) ** 2
    if reduce:
        return torch.mean(l)
    return l

def weighted_direction_loss(gt_directions: Tensor, gt_z_val: Tensor, pred_directions: Tensor, transmittance: Tensor, near: float, far: float, loss_type: Literal[-1, 0, 1] = -1, predict_z_val_type: Literal[-1, 0, 1] = -1) -> Tensor:
    direction_loss = 0
    # import ipdb; ipdb.set_trace()
    if predict_z_val_type != -1:
        pred_directions, pred_z_val = torch.split(pred_directions, [3, 1], -1)
        if predict_z_val_type == 0:
            z_val_diff = _img2mse(pred_z_val, gt_z_val, reduce=False)[..., 0].amin(-1)
            direction_loss += z_val_diff
        elif predict_z_val_type == 1:
            loss_type = 0
            gt_directions = gt_directions * gt_z_val
            pred_directions = pred_directions * pred_z_val
        elif predict_z_val_type == 2:
            pred_z_val = pred_z_val * (far - near) + near
            z_val_diff = _img2mse(pred_z_val, gt_z_val, reduce=False)[..., 0].amin(-1)
            direction_loss += z_val_diff
    if loss_type == -1:
        # transmittance weighting is one the loss
        direction_loss = dir_dot_loss(gt_directions, pred_directions, reduce=False).amin(-1, keepdim=True)
        direction_loss = direction_loss * transmittance[..., 0]
    elif loss_type == 0:
        # mse loss with trans weighted gt direction
        gt_directions = gt_directions * transmittance
        direction_loss = direction_loss + _img2mse(gt_directions, pred_directions, reduce=False).mean(-1).amin(-1)
    elif loss_type == 1:
        # trans and dir loss separate
        pred_norm = torch.norm(pred_directions, p=2, dim=-1, keepdim=True)
        norm_loss = _img2mse(pred_norm, transmittance, reduce=False)[..., 0]
        pred_directions = F.normalize(pred_directions, dim=-1)
        direction_loss = dir_dot_loss(gt_directions, pred_directions, reduce=False)
        direction_loss = direction_loss + (direction_loss + norm_loss).amin(-1)
    return direction_loss