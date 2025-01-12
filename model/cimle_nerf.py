import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import Optional
from model.run_nerf_helpers import DenseLayer
from model.mlp import MLP, Constant
from functools import partial
import math

def _calculate_fan_in_and_fan_out(tensor: Tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

def xavier_normal_(weight: Tensor, gain=1, seed=None):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(weight)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))

    return _no_grad_normal_(weight, 0., std, seed=seed)

def _no_grad_normal_(tensor: Tensor, mean: float, std: float, seed: Optional[int]=None):
    with torch.no_grad():
        if seed is not None:
            gen = torch.Generator(device=tensor.device).manual_seed(seed)
        else:
            gen = None
        return tensor.normal_(mean, std, generator=gen)

# Model
class cIMLENeRF(nn.Module):
    def __init__(
        self, 
        D=8, 
        W=256, 
        input_ch=3, 
        input_ch_views=3, 
        input_ch_cam=0, 
        output_ch=4, 
        skips=[4], 
        use_viewdirs=False,
        
        input_ch_rand=3,
        cimle_sample_num=32, 
        cimle_latent_dim=32, 
        gain_factor=math.sqrt(2), 
        normalize_output_dir=True,
        include_input_in_dir_ch=True,
        trans_pred_type=-1, 
        direction_num_layers=2,
        random_fn_num_layers=1,
        use_bias=False,
        predict_z_val_type = 0):
        """ 
        """
        super(cIMLENeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_cam = input_ch_cam
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        assert use_viewdirs
        
        self.trans_pred_type = trans_pred_type
        self.input_ch_rand=input_ch_rand
        self.cimle_sample_num = cimle_sample_num
        self.cimle_latent_dim = cimle_latent_dim
        self.gain_factor=gain_factor
        self.normalize_output_dir=normalize_output_dir
        self.include_input_in_dir_ch=include_input_in_dir_ch
        self.predict_z_val_type=predict_z_val_type
        
        self.pts_linears = nn.ModuleList(
            [DenseLayer(input_ch, W, activation="relu")] + [DenseLayer(W, W, activation="relu") if i not in self.skips else DenseLayer(W + input_ch, W, activation="relu") for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([DenseLayer(input_ch_views + input_ch_cam + W, W//2, activation="relu")])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = DenseLayer(W, W, activation="linear")
            self.alpha_linear = DenseLayer(W, 1, activation="linear")
            self.rgb_linear = DenseLayer(W//2, 3, activation="linear")
            self.random_linears = MLP(input_ch_rand, random_fn_num_layers, W, out_dim=cimle_latent_dim, activation=nn.ReLU(), use_bias=use_bias, group=self.cimle_sample_num)
            out_dim = 3 + int(predict_z_val_type != -1) + int(self.trans_pred_type == 0)
            self.direction_linears = MLP(W + cimle_latent_dim + int(self.include_input_in_dir_ch) * self.input_ch_rand, direction_num_layers, W, out_dim=out_dim, activation=nn.ReLU())
        else:
            self.output_linear = DenseLayer(W, output_ch, activation="linear")
        print(self)
            
    def load_state_dict_from_vanilla(self, state_dict, freeze_pretrained=False):
        state_dict = {k.replace("NeRF", "cIMLENeRF"): v for k, v in state_dict.items()}
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        load_keys = [k for k in state_dict.keys() if "direction_linears" not in k and "random_linears" not in k]
        state_dict.update({k: v for k, v in self.state_dict().items() if k not in load_keys})
        self.load_state_dict(state_dict)
        for n, p in self.named_parameters():
            if n in load_keys:
                p.requires_grad = not freeze_pretrained

    def load_cimle_weights(self):
        print("Loading weights")
        self.set_random_fns_weights(self._heldout_rand_fn_weights)
        self._heldout_rand_fn_weights=None
    
    def save_cimle_weights(self):
        print("Saving weights")
        self._heldout_rand_fn_weights = self.get_random_fns_weights()

        
    def sample_random_fns(self, seed=None):
        self.random_linears.requires_grad = False
        self.random_linears.apply(partial(self._sample_weights, seed=seed))
    
    def get_random_fns_weights(self):
        return self.random_linears.state_dict()
    
    def set_random_fns_weights(self, state_dict):
        if state_dict is None:
            return 
        self.random_linears.load_state_dict(state_dict=state_dict)
        self.random_linears.requires_grad = False 
        
    def get_param_group(self):
        normal_group = []
        cimle_group = []
        for name, p in self.named_parameters():
            if "direction_linears" in name:
                cimle_group.append((name, p))
            elif "random_linears" not in name:
                normal_group.append((name, p))
        return normal_group, cimle_group
    
    def get_random_fns_modules(self):
        return self.random_linears
    
    def _sample_weights(self, m, seed=None):
        if isinstance(m, nn.Linear):
            if hasattr(m, "weight"):
                xavier_normal_(m.weight, gain=self.gain_factor, seed=seed)
            if getattr(m, "bias", None) is not None:
                _no_grad_normal_(m.bias, 0, math.sqrt(1 / self.input_ch_rand) * self.gain_factor, seed=seed)
        elif isinstance(m, nn.Conv1d):
            if hasattr(m, "weight"):
                _no_grad_normal_(m.weight, 0, math.sqrt(self.cimle_sample_num / m.in_channels) * self.gain_factor, seed=seed)
            if getattr(m, "bias", None) is not None:
                _no_grad_normal_(m.bias, 0, math.sqrt(self.cimle_sample_num / m.in_channels) * self.gain_factor, seed=seed)
        elif isinstance(m, Constant):
            _no_grad_normal_(m.bias, 0, math.sqrt(self.cimle_sample_num / m.out_dim) * self.gain_factor, seed=seed)


    def forward(self, input_pts: Tensor, input_pts_rand: Optional[Tensor] = None, input_views: Optional[Tensor] = None):
        assert input_pts.shape[-1] == self.input_ch
        
        if input_pts_rand is not None:
            assert input_pts_rand.shape[-1] == self.input_ch_rand
        if input_views is not None:
            assert input_views.shape[-1] == self.input_ch_views + self.input_ch_cam

        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)
        
        if input_pts_rand is not None:
            direction_pred = self.get_directions(feature, input_pts_rand)
        else:
            direction_pred = torch.zeros_like(alpha).unsqueeze(-2)
        
        if input_views is not None:
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
        else:
            rgb = torch.zeros(list(alpha.shape[:-1]) + [3]).to(input_pts.device)
            
        outputs = torch.cat([rgb, F.softplus(alpha, beta=10)], -1)

        return outputs, direction_pred

    
    def get_directions(self, feature: Tensor, input_pts_rand: Tensor):
        sh = input_pts_rand.shape[:-1]
        cimle_latents = self.random_linears(
            input_pts_rand.unsqueeze(-2).repeat_interleave(self.cimle_sample_num, -2).reshape(*sh, self.input_ch_rand * self.cimle_sample_num)
            ).reshape(list(input_pts_rand.shape[:-1]) + [self.cimle_sample_num, self.cimle_latent_dim])
        hd = torch.cat([feature.unsqueeze(-2).repeat_interleave(self.cimle_sample_num, -2), cimle_latents], dim=-1) # [..., cimle_sample_num, W + cimle_latent_dim]
        if self.include_input_in_dir_ch:
            hd = torch.cat([input_pts_rand.unsqueeze(-2).repeat_interleave(self.cimle_sample_num, -2), hd], dim=-1)
        
        direction_branch_out = self.direction_linears(hd)
        if self.predict_z_val_type != -1:
            direction_pred, z_pred = direction_branch_out[..., :-1], direction_branch_out[..., -1:]
            if self.predict_z_val_type == 2:
                z_pred = F.sigmoid(z_pred)
            elif self.predict_z_val_type in [0, 1]:
                z_pred = F.relu(z_pred)
        else:
            direction_pred = direction_branch_out
        if self.normalize_output_dir:
            direction_pred = F.normalize(direction_pred, dim=-1)
        else:
            if self.trans_pred_type == 0:
                assert direction_pred.shape[-1] == 4
                direction_pred, trans_pred = direction_pred.split([3, 1], -1)
                direction_pred = F.normalize(direction_pred, dim=-1)
                trans_pred = F.sigmoid(trans_pred)
                direction_pred = trans_pred * direction_pred
                
        if self.predict_z_val_type != -1:
            direction_pred = torch.cat([direction_pred, z_pred], -1)
            assert direction_pred.shape[-1] == 4
        else:
            assert direction_pred.shape[-1] == 3
        return direction_pred