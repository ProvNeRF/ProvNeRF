'''
Mikaela Uy
mikacuy@stanford.edu
For Scannet data
Modified from DDP codebase
'''
import os
import shutil
import subprocess
import math
import time
import datetime
from argparse import Namespace

import configargparse
from skimage.metrics import structural_similarity
from lpips import LPIPS
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter
from model.cimle_nerf import cIMLENeRF
from metric.direction_metrics import *
from tqdm import trange
from pathlib import Path

from model import NeRF, get_embedder, get_rays, sample_pdf, sample_pdf_joint, img2mse, mse2psnr, to8b, \
    compute_depth_loss, select_coordinates, to16b, compute_space_carving_loss, \
    sample_pdf_return_u, sample_pdf_joint_return_u
from data import load_scene_scannet
from train_utils import MeanTracker, update_learning_rate, get_learning_rate
from metric import compute_rmse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = False

def batchify_double(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs, embedded_rand, embedded_dirs):
        for i in range(0, inputs.shape[0], chunk):
            t0 = time.time()
            embedded_rand_chunk = None if embedded_rand is None else embedded_rand[i:i+chunk]
            embedded_dirs_chunk = None if embedded_dirs is None else embedded_dirs[i:i+chunk]
            a, b = fn(inputs[i:i+chunk], embedded_rand_chunk, embedded_dirs_chunk)
            # print('single forward time:',time.time()-t0)
            if i == 0:
                A = a
                B = b
            else:
                A = torch.cat([A,a],dim=0)
                B = torch.cat([B,b],dim=0)
        return A, B
    return ret


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

def run_network(iteration, inputs, viewdirs, embedded_cam, fn, embed_fn, embeddirs_fn, embedrand_fn, bb_center, bb_scale, netchunk=1024*64, num_iterations=1,ret_dirs=True):
    """Prepares inputs and applies network 'fn'.
    """
    if isinstance(fn.module, NeRF):
        return run_network_vanilla(inputs, viewdirs, embedded_cam, fn, embed_fn, embeddirs_fn, bb_center, bb_scale, netchunk)
    elif isinstance(fn.module, cIMLENeRF):
        return run_network_cimle(iteration, inputs, viewdirs, embedded_cam, fn, embed_fn, embeddirs_fn, embedrand_fn, bb_center, bb_scale, netchunk, num_iterations, ret_dirs)

def run_network_vanilla(inputs, viewdirs, embedded_cam, fn, embed_fn, embeddirs_fn, bb_center, bb_scale, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    inputs_flat = (inputs_flat - bb_center) * bb_scale
    embedded = embed_fn(inputs_flat) # samples * rays, multires * 2 * 3 + 3

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs, embedded_cam.unsqueeze(0).expand(embedded_dirs.shape[0], embedded_cam.shape[0])], -1)
        
    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

def run_network_cimle(iteration, inputs, viewdirs, embedded_cam, fn, embed_fn, embeddirs_fn, embedrand_fn, bb_center=0, bb_scale=1, netchunk=1024*64, num_iterations=1, ret_dirs=True):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    inputs_flat = (inputs_flat - bb_center) * bb_scale
    embedded = embed_fn(inputs_flat) # samples * rays, multires * 2 * 3 + 3
    L = embedded.shape[1]
    if num_iterations > 0:
        embedded[:, min(int(iteration * (L-3) / num_iterations), L-3)+3:]=0
    if ret_dirs:
        embedded_rand = embedrand_fn(inputs_flat)
    else:
        embedded_rand = None

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded_dirs = torch.cat([embedded_dirs, embedded_cam.unsqueeze(0).expand(embedded_dirs.shape[0], embedded_cam.shape[0])], -1)
    else:
        embedded_dirs = None
        
    outputs_flat, directions_flat = batchify_double(fn, netchunk)(embedded, embedded_rand, embedded_dirs)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    directions = torch.reshape(directions_flat, list(inputs.shape[:-1]) + list(directions_flat.shape[-2:]))
    return outputs, directions
    
    


def batchify_rays(rays_flat, chunk=1024*32, use_viewdirs=False, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], use_viewdirs, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret

def render(H, W, intrinsic, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1., with_5_9=False, use_viewdirs=False, c2w_staticcam=None, ret_dirs=True,
                  rays_depth=None, **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      with_5_9: render with aspect ratio 5.33:9 (one third of 16:9)
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, intrinsic, c2w)
        if with_5_9:
            W_before = W
            W = int(H / 9. * 16. / 3.)
            if W % 2 != 0:
                W = W - 1
            start = (W_before - W) // 2
            rays_o = rays_o[:, start:start + W, :]
            rays_d = rays_d[:, start:start + W, :]
    elif rays.shape[0] == 2:
        # use provided ray batch
        rays_o, rays_d = rays
    else:
        rays_o, rays_d, rays_depth = rays
    
    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, intrinsic, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)
    if rays_depth is not None:
        rays_depth = torch.reshape(rays_depth, [-1,3]).float()
        rays = torch.cat([rays, rays_depth], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, use_viewdirs, ret_dirs=ret_dirs, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

def render_hyp(H, W, intrinsic, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1., with_5_9=False, use_viewdirs=False, c2w_staticcam=None, 
                  rays_depth=None, **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      with_5_9: render with aspect ratio 5.33:9 (one third of 16:9)
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for 
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, intrinsic, c2w)
        if with_5_9:
            W_before = W
            W = int(H / 9. * 16. / 3.)
            if W % 2 != 0:
                W = W - 1
            start = (W_before - W) // 2
            rays_o = rays_o[:, start:start + W, :]
            rays_d = rays_d[:, start:start + W, :]
    elif rays.shape[0] == 2:
        # use provided ray batch
        rays_o, rays_d = rays
    else:
        rays_o, rays_d, rays_depth = rays
    
    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, intrinsic, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)

    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)
    if rays_depth is not None:
        rays_depth = torch.reshape(rays_depth, [-1,3]).float()
        rays = torch.cat([rays, rays_depth], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, use_viewdirs, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_video(poses, H, W, intrinsics, filename, args, render_kwargs_test, fps=25):
    video_dir = os.path.join(args.ckpt_dir, args.expname, 'video_' + filename)
    if os.path.exists(video_dir):
        shutil.rmtree(video_dir)
    os.makedirs(video_dir, exist_ok=True)
    depth_scale = render_kwargs_test["far"]
    max_depth_in_video = 0
    for img_idx in trange(0, len(poses), 3):
    # for img_idx in range(200):
        pose = poses[img_idx, :3,:4]
        intrinsic = intrinsics[img_idx, :]
        with torch.no_grad():
            if args.input_ch_cam > 0:
                render_kwargs_test["embedded_cam"] = torch.zeros((args.input_ch_cam), device=device)
            # render video in 16:9 with one third rgb, one third depth and one third depth standard deviation
            rgb, _, _, extras = render(H, W, intrinsic, chunk=args.N_rand, c2w=pose, ret_dirs=False, **render_kwargs_test)
            rgb_cpu_numpy_8b = to8b(rgb.cpu().numpy())
            video_frame = cv2.cvtColor(rgb_cpu_numpy_8b, cv2.COLOR_RGB2BGR)
            max_depth_in_video = max(max_depth_in_video, extras['depth_map'].max())
            depth_frame = cv2.applyColorMap(to8b((extras['depth_map'] / depth_scale).cpu().numpy()), cv2.COLORMAP_TURBO)
            video_frame = np.concatenate((video_frame, depth_frame), 1)
            depth_var = ((extras['z_vals'] - extras['depth_map'].unsqueeze(-1)).pow(2) * extras['weights']).sum(-1)
            depth_std = depth_var.clamp(0., 1.).sqrt()
            video_frame = np.concatenate((video_frame, cv2.applyColorMap(to8b(depth_std.cpu().numpy()), cv2.COLORMAP_VIRIDIS)), 1)
            cv2.imwrite(os.path.join(video_dir, str(img_idx) + '.jpg'), video_frame)

    video_file = os.path.join(args.ckpt_dir, args.expname, filename + '.mp4')
    subprocess.call(["ffmpeg", "-y", "-framerate", str(fps), "-i", os.path.join(video_dir, "%d.jpg"), "-c:v", "libx264", "-profile:v", "high", "-crf", str(fps), video_file])
    print("Maximal depth in video: {}".format(max_depth_in_video))

def render_images_with_metrics(count, indices, images, depths, valid_depths, poses, H, W, intrinsics, lpips_alex, args, render_kwargs_test, \
    embedcam_fn=None, with_test_time_optimization=False):
    far = render_kwargs_test['far']

    if count is None:
        # take all images in order
        count = len(indices)
        img_i = indices
    else:
        # take random images
        img_i = np.random.choice(indices, size=count, replace=False)

    rgbs_res = torch.empty(count, 3, H, W)
    rgbs0_res = torch.empty(count, 3, H, W)
    target_rgbs_res = torch.empty(count, 3, H, W)
    depths_res = torch.empty(count, 1, H, W)
    depths0_res = torch.empty(count, 1, H, W)
    target_depths_res = torch.empty(count, 1, H, W)
    target_valid_depths_res = torch.empty(count, 1, H, W, dtype=bool)
    
    mean_metrics = MeanTracker()
    mean_depth_metrics = MeanTracker() # track separately since they are not always available
    for n, img_idx in enumerate(img_i):
        print("Render image {}/{}".format(n + 1, count), end="")
        target = images[img_idx]
        target_depth = depths[img_idx]
        target_valid_depth = valid_depths[img_idx]
        pose = poses[img_idx, :3,:4]
        intrinsic = intrinsics[img_idx, :]

        with torch.no_grad():
            rgb, _, _, extras = render(H, W, intrinsic, chunk=args.N_rand, c2w=pose, ret_dirs=False, **render_kwargs_test)
            
            # compute depth rmse
            depth_rmse = compute_rmse(extras['depth_map'][target_valid_depth], target_depth[:, :, 0][target_valid_depth])
            if not torch.isnan(depth_rmse):
                depth_metrics = {"depth_rmse" : depth_rmse.item()}
                mean_depth_metrics.add(depth_metrics)

            # ### Fit LSTSQ for white balancing
            # rgb_reshape = rgb.view(1, -1, 3)
            # target_reshape = target.view(1, -1, 3)

            # ## No intercept          
            # X = torch.linalg.lstsq(rgb_reshape, target_reshape).solution
            # rgb_reshape = rgb_reshape @ X
            # rgb_reshape = rgb_reshape.view(rgb.shape)
            # rgb = rgb_reshape
            
            # compute color metrics
            img_loss = img2mse(rgb, target)
            psnr = mse2psnr(img_loss)
            print("PSNR: {}".format(psnr))
            rgb = torch.clamp(rgb, 0, 1)
            ssim = structural_similarity(rgb.cpu().numpy(), target.cpu().numpy(), data_range=1., channel_axis=-1)
            lpips = lpips_alex(rgb.permute(2, 0, 1).unsqueeze(0), target.permute(2, 0, 1).unsqueeze(0), normalize=True)[0]
            
            # store result
            rgbs_res[n] = rgb.clamp(0., 1.).permute(2, 0, 1).cpu()
            target_rgbs_res[n] = target.permute(2, 0, 1).cpu()
            depths_res[n] = (extras['depth_map'] / far).unsqueeze(0).cpu()
            target_depths_res[n] = (target_depth[:, :, 0] / far).unsqueeze(0).cpu()
            target_valid_depths_res[n] = target_valid_depth.unsqueeze(0).cpu()
            metrics = {"img_loss" : img_loss.item(), "psnr" : psnr.item(), "ssim" : ssim, "lpips" : lpips[0, 0, 0],}
            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target)
                psnr0 = mse2psnr(img_loss0)
                depths0_res[n] = (extras['depth0'] / far).unsqueeze(0).cpu()
                rgbs0_res[n] = torch.clamp(extras['rgb0'], 0, 1).permute(2, 0, 1).cpu()
                metrics.update({"img_loss0" : img_loss0.item(), "psnr0" : psnr0.item()})
            mean_metrics.add(metrics)
    
    res = { "rgbs" :  rgbs_res, "target_rgbs" : target_rgbs_res, "depths" : depths_res, "target_depths" : target_depths_res, \
        "target_valid_depths" : target_valid_depths_res}
    if 'rgb0' in extras:
        res.update({"rgbs0" : rgbs0_res, "depths0" : depths0_res,})
    all_mean_metrics = MeanTracker()
    all_mean_metrics.add({**mean_metrics.as_dict(), **mean_depth_metrics.as_dict()})
    return all_mean_metrics, res

def write_images_with_metrics(images, mean_metrics, far, args, with_test_time_optimization=False, step=None):
    result_dir = os.path.join(args.ckpt_dir, args.expname, "test_images_" + ("with_optimization_" if with_test_time_optimization else "") + args.scene_id + ("" if step is None else f"{step:09d}"))
    os.makedirs(result_dir, exist_ok=True)
    for n, (rgb, depth) in enumerate(zip(images["rgbs"].permute(0, 2, 3, 1).cpu().numpy(), \
            images["depths"].permute(0, 2, 3, 1).cpu().numpy())):

        # write rgb
        cv2.imwrite(os.path.join(result_dir, str(n) + "_rgb" + ".jpg"), cv2.cvtColor(to8b(rgb), cv2.COLOR_RGB2BGR))
        # write depth
        cv2.imwrite(os.path.join(result_dir, str(n) + "_d" + ".png"), to16b(depth))

    with open(os.path.join(result_dir, 'metrics.txt'), 'w') as f:
        mean_metrics.print(f)
    mean_metrics.print()

def load_checkpoint(args, ret_all=False):
    path = os.path.join(args.ckpt_dir, args.expname)
    ckpts = [os.path.join(path, f) for f in sorted(os.listdir(path)) if '000.tar' in f]
    print('Found ckpts', ckpts)
    ckpt = None
    if ret_all:
        return ckpts
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)
    return ckpt

def create_nerf(args, scene_render_params, ft_path=None):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    embed_rand_fn, input_ch_dir_br = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]

    model = NeRF(D=args.netdepth, W=args.netwidth, 
                     input_ch=input_ch, output_ch=output_ch, skips=skips,
                     input_ch_views=input_ch_views, input_ch_cam=args.input_ch_cam, use_viewdirs=args.use_viewdirs)

    model = nn.DataParallel(model).to(device)
    grad_vars = list(model.parameters())

    grad_vars = []
    grad_vars_cimle = []
    grad_names = []
    grad_names_cimle = []

    for name, param in model.named_parameters():
        grad_vars.append(param)
        grad_names.append(name)


    model_fine = None
    if args.N_importance > 0:
        model_fine = cIMLENeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, input_ch_cam=args.input_ch_cam, 
                          use_viewdirs=args.use_viewdirs, 
                          
                          input_ch_rand=input_ch_dir_br, cimle_latent_dim=args.cimle_latent_dim, cimle_sample_num=args.cimle_sample_num,
                          gain_factor=math.sqrt(2), normalize_output_dir=args.direction_loss_type == -1,
                          include_input_in_dir_ch=True, direction_num_layers=args.direction_num_layers, trans_pred_type=-1,
                          random_fn_num_layers=args.random_fn_num_layers, use_bias=args.use_bias, predict_z_val_type=args.predict_z_val_type)
            
        model_fine = nn.DataParallel(model_fine).to(device)
        fine_grad_vars, cimle_grad_vars = model_fine.module.get_param_group()
        for name, param in fine_grad_vars:
            grad_vars.append(param)
            grad_names.append(name)
        
        for name, param in cimle_grad_vars:
            grad_vars_cimle.append(param)
            grad_names_cimle.append(name)

    network_query_fn = lambda iteration, inputs, viewdirs, embedded_cam, network_fn, ret_dirs=True: run_network(iteration, inputs, viewdirs, embedded_cam, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                embedrand_fn=embed_rand_fn,
                                                                bb_center=args.bb_center,
                                                                bb_scale=args.bb_scale,
                                                                num_iterations=int(args.num_iterations * args.reg_percent),
                                                                netchunk=args.netchunk_per_gpu*args.n_gpus,
                                                                ret_dirs=ret_dirs)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    optimizer_cimle = torch.optim.Adam(params=grad_vars_cimle, lr=args.cimle_lrate, betas=(0.9, 0.999))


    start = 0

    ##########################
    # Load checkpoints
    ckpt = load_checkpoint(args)
    if ft_path is not None:
        print('Reloading from ft_path', ft_path)
        ckpt = torch.load(ft_path)
    if ckpt is not None:
        start = ckpt['global_step']
        # optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
            
        

    ##########################
    embedded_cam = torch.tensor((), device=device)
    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'embedded_cam' : embedded_cam,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'use_viewdirs' : args.use_viewdirs,
        'raw_noise_std' : args.raw_noise_std,
        "M" : args.floater_M
    }
    render_kwargs_train.update(scene_render_params)

    render_kwargs_train['ndc'] = False
    render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    
    

    if args.restart and ft_path is None:
        start = 0
    
    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, grad_names, grad_vars_cimle, optimizer_cimle, grad_names_cimle

def compute_weights(raw, z_vals, rays_d, noise=0., ret_trans=False):
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-(act_fn(raw)*dists))

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.full_like(dists[...,:1], 1e10, device=device)], -1)  # [N_rays, N_samples]
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    trans = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    weights = alpha * trans
    if ret_trans:
        return weights, trans
    return weights

def raw2depth(raw, z_vals, rays_d):
    weights = compute_weights(raw, z_vals, rays_d)
    depth = torch.sum(weights * z_vals, -1)
    std = (((z_vals - depth.unsqueeze(-1)).pow(2) * weights).sum(-1)).sqrt()
    return depth, std

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    weights, transmittance = compute_weights(raw, z_vals, rays_d, noise, ret_trans=True)
    
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    return rgb_map, disp_map, acc_map, weights, depth_map, transmittance

def perturb_z_vals(z_vals, pytest):
    # get intervals between samples
    mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
    upper = torch.cat([mids, z_vals[...,-1:]], -1)
    lower = torch.cat([z_vals[...,:1], mids], -1)
    # stratified samples in those intervals
    t_rand = torch.rand_like(z_vals)

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        t_rand = np.random.rand(*list(z_vals.shape))
        t_rand = torch.Tensor(t_rand)

    z_vals = lower + (upper - lower) * t_rand
    return z_vals

def render_rays(ray_batch,
                use_viewdirs,
                network_fn,
                network_query_fn,
                N_samples,
                iteration=None,
                precomputed_z_samples=None,
                embedded_cam=None,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                raw_noise_std=0.,
                ret_dirs=True,
                verbose=False,
                pytest=False,
                M=None,
                is_joint=False,
                cached_u= None):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = None
    depth_range = None
    if use_viewdirs:
        viewdirs = ray_batch[:,8:11]
        if ray_batch.shape[-1] > 11:
            depth_range = ray_batch[:,11:14]
    else:
        if ray_batch.shape[-1] > 8:
            depth_range = ray_batch[:,8:11]
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]
    t_vals = torch.linspace(0., 1., steps=N_samples)

    # sample and render rays for dense depth priors for nerf
    N_samples_half = N_samples // 2
    
    # sample and render rays for nerf
    if not lindisp:
        # print("Not lindisp")
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        # print("Lindisp")
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    if perturb > 0.:
        # print("Perturb.")
        z_vals = perturb_z_vals(z_vals, pytest)

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

    raw = network_query_fn(iteration, pts, viewdirs, embedded_cam, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map, _ = raw2outputs(raw, z_vals, rays_d, raw_noise_std, pytest=pytest)


    ### Try without coarse and fine network, but just one network and use additional samples from the distribution of the nerf
    if N_importance == 0:

        ### P_depth from base network
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])

        if not is_joint:
            z_vals_2 = sample_pdf(z_vals_mid, weights[...,1:-1], N_samples, det=(perturb==0.), pytest=pytest)
        else:
            z_vals_2 = sample_pdf_joint(z_vals_mid, weights[...,1:-1], N_samples, det=(perturb==0.), pytest=pytest)
        #########################

        ### Forward the rendering network with the additional samples
        pts_2 = rays_o[...,None,:] + rays_d[...,None,:] * z_vals_2[...,:,None]
        raw_2 = network_query_fn(pts_2, viewdirs, embedded_cam, network_fn)
        z_vals = torch.cat((z_vals, z_vals_2), -1)
        raw = torch.cat((raw, raw_2), 1)
        z_vals, indices = z_vals.sort()

        ### Concatenated output
        raw = torch.gather(raw, 1, indices.unsqueeze(-1).expand_as(raw))
        rgb_map, disp_map, acc_map, weights, depth_map, _ = raw2outputs(raw, z_vals, rays_d, raw_noise_std, pytest=pytest)


        ## Second tier P_depth
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])

        if not is_joint:
            z_vals_output = sample_pdf(z_vals_mid, weights[...,1:-1], N_samples, det=(perturb==0.), pytest=pytest)
        else:
            z_vals_output = sample_pdf_joint(z_vals_mid, weights[...,1:-1], N_samples, det=(perturb==0.), pytest=pytest)

        pred_depth_hyp = torch.cat((z_vals_2, z_vals_output), -1)


    elif N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0, depth_map_0, z_vals_0, weights_0 = rgb_map, disp_map, acc_map, depth_map, z_vals, weights

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])

        ## Original NeRF uses this
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        
        ## To model p_depth from coarse network
        z_samples_depth = torch.clone(z_samples)

        ## For fine network sampling
        z_samples = z_samples.detach()

        z_vals, z_val_sort_idxes = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        fine_index_mask = torch.arange(z_vals.shape[-1]).expand_as(z_vals) >= N_samples
        fine_index_mask = torch.gather(fine_index_mask, -1, z_val_sort_idxes)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine

        raw, directions = network_query_fn(iteration, pts, viewdirs, embedded_cam, run_fn, ret_dirs=ret_dirs)

        rgb_map, disp_map, acc_map, weights, depth_map, transmittance = raw2outputs(raw, z_vals, rays_d, raw_noise_std, pytest=pytest)

        if M is not None:
            mask = torch.ones(N_samples+N_importance, device=device)
            mask[M:] = 0
            floater_loss_fine = (raw[...,3] * mask).mean(1)

        ### P_depth from fine network
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])

        if not is_joint:
            z_samples, u = sample_pdf_return_u(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest, load_u=cached_u)
        else:
            z_samples, u = sample_pdf_joint_return_u(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest, load_u=cached_u)

        pred_depth_hyp = z_samples


    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'depth_map' : depth_map, 'z_vals' : z_vals, 'weights' : weights, 'pred_hyp' : pred_depth_hyp,\
    'u':u, "trans":transmittance, "directions":directions, "fine_index_mask":fine_index_mask, "pts": pts}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['depth0'] = depth_map_0
        ret['z_vals0'] = z_vals_0
        ret['weights0'] = weights_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
        # ret['pred_hyp'] = pred_depth_hyp

    if M is not None:
        ret["floater_loss_coarse"] = floater_loss_coarse
        if N_importance > 0:
            ret["floater_loss_fine"] = floater_loss_fine
        # ret['pred_hyp'] = pred_depth_hyp


    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret

def get_ray_batch_from_one_image(H, W, i_train, images, depths, valid_depths, poses, intrinsics, args):
    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W), indexing='ij'), -1)  # (H, W, 2)
    img_i = np.random.choice(i_train)
    target = images[img_i]
    target_depth = depths[img_i]
    target_valid_depth = valid_depths[img_i]
    pose = poses[img_i]
    intrinsic = intrinsics[img_i, :]
    rays_o, rays_d = get_rays(H, W, intrinsic, pose)  # (H, W, 3), (H, W, 3)
    select_coords = select_coordinates(coords, args.N_rand)
    rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    target_d = target_depth[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 1) or (N_rand, 2)
    target_vd = target_valid_depth[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 1)

    batch_rays = torch.stack([rays_o, rays_d], 0)  # (2, N_rand, 3)
    return batch_rays, target_s, target_d, target_vd, img_i

def get_ray_batch_from_one_image_hypothesis_idx(H, W, img_i, images, depths, valid_depths, poses, intrinsics, all_hypothesis, args, space_carving_idx=None, cached_u=None):
    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W), indexing='ij'), -1)  # (H, W, 2)
    # img_i = np.random.choice(i_train)
    
    target = images[img_i]
    target_depth = depths[img_i]
    target_valid_depth = valid_depths[img_i]
    pose = poses[img_i]
    intrinsic = intrinsics[img_i, :]

    target_hypothesis = all_hypothesis[img_i]

    rays_o, rays_d = get_rays(H, W, intrinsic, pose)  # (H, W, 3), (H, W, 3)
    select_coords = select_coordinates(coords, args.N_rand)
    rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
    target_d = target_depth[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 1) or (N_rand, 2)
    target_vd = target_valid_depth[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 1)
    target_h = target_hypothesis[:, select_coords[:, 0], select_coords[:, 1]]

    if space_carving_idx is not None:
        # print(space_carving_idx.shape)
        # print(space_carving_idx[img_i, select_coords[:, 0], select_coords[:, 1]].shape)
        target_hypothesis  = target_hypothesis.repeat(1, 1, 1, space_carving_idx.shape[-1])

        curr_space_carving_idx = space_carving_idx[img_i, select_coords[:, 0], select_coords[:, 1]]

        target_h_rays = target_hypothesis[ :, select_coords[:, 0], select_coords[:, 1]]

        target_h = torch.gather(target_h_rays, 1, curr_space_carving_idx.unsqueeze(0).long())


    if cached_u is not None:
        curr_cached_u = cached_u[img_i, select_coords[:, 0], select_coords[:, 1]]
    else:
        curr_cached_u = None

    if args.mask_corners:
        ### Initialize a masked image
        space_carving_mask = torch.ones((target.shape[0], target.shape[1]), dtype=torch.float, device=images.device)

        ### Mask out the corners
        num_pix_to_mask = 20
        space_carving_mask[:num_pix_to_mask, :num_pix_to_mask] = 0
        space_carving_mask[:num_pix_to_mask, -num_pix_to_mask:] = 0
        space_carving_mask[-num_pix_to_mask:, :num_pix_to_mask] = 0
        space_carving_mask[-num_pix_to_mask:, -num_pix_to_mask:] = 0

        space_carving_mask = space_carving_mask[select_coords[:, 0], select_coords[:, 1]]
    else:
        space_carving_mask = None

    batch_rays = torch.stack([rays_o, rays_d], 0)  # (2, N_rand, 3)
    
    return batch_rays, target_s, target_d, target_vd, img_i, target_h, space_carving_mask, curr_cached_u


def train_nerf(images, depths, valid_depths, poses, intrinsics, i_split, args, scene_sample_params, lpips_alex, gt_depths, gt_valid_depths, all_depth_hypothesis, is_init_scales=False, scales_init=None, shifts_init=None):
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    tb = SummaryWriter(log_dir=os.path.join(args.ckpt_dir, args.expname))
    near, far = scene_sample_params['near'], scene_sample_params['far']
    H, W = images.shape[1:3]
    i_train, i_val, i_test, i_video = i_split
    print('TRAIN views are', i_train)
    print('VAL views are', i_val)
    print('TEST views are', i_test)

    # use ground truth depth for validation and test if available
    if gt_depths is not None:
        depths[i_test] = gt_depths[i_test]
        valid_depths[i_test] = gt_valid_depths[i_test]
        depths[i_val] = gt_depths[i_val]
        valid_depths[i_val] = gt_valid_depths[i_val]
    i_relevant_for_training = np.concatenate((i_train, i_val), 0)
    if len(i_test) == 0:
        print("Error: There is no test set")
        exit()
    if len(i_val) == 0:
        print("Warning: There is no validation set, test set is used instead")
        i_val = i_test
        i_relevant_for_training = np.concatenate((i_relevant_for_training, i_val), 0)

    # keep test data on cpu until needed
    test_images = images[i_test]
    test_depths = depths[i_test]
    test_valid_depths = valid_depths[i_test]
    test_poses = poses[i_test]
    test_intrinsics = intrinsics[i_test]
    i_test = i_test - i_test[0]

    # move training data to gpu
    images = torch.Tensor(images[i_relevant_for_training]).to(device)
    depths = torch.Tensor(depths[i_relevant_for_training]).to(device)
    valid_depths = torch.Tensor(valid_depths[i_relevant_for_training]).bool().to(device)
    poses = torch.Tensor(poses[i_relevant_for_training]).to(device)
    intrinsics = torch.Tensor(intrinsics[i_relevant_for_training]).to(device)
    all_depth_hypothesis = torch.Tensor(all_depth_hypothesis).to(device)
    intrinsics = torch.Tensor(intrinsics[i_relevant_for_training]).to(device)

    # create nerf model
    render_kwargs_train, render_kwargs_test, start, nerf_grad_vars, optimizer, nerf_grad_names, cimle_nerf_grad_vars, optimizer_cimle, cimle_nerf_grad_names = create_nerf(args, scene_sample_params)
    
    ##### Initialize depth scale and shift
    DEPTH_SCALES = torch.autograd.Variable(torch.ones((images.shape[0], 1), dtype=torch.float, device=images.device)*args.scale_init, requires_grad=True)
    DEPTH_SHIFTS = torch.autograd.Variable(torch.ones((images.shape[0], 1), dtype=torch.float, device=images.device)*args.shift_init, requires_grad=True)

    print(DEPTH_SCALES)
    print()
    print(DEPTH_SHIFTS)
    print()
    print(DEPTH_SCALES.shape)
    print(DEPTH_SHIFTS.shape)

    optimizer_ss = torch.optim.Adam(params=(DEPTH_SCALES, DEPTH_SHIFTS,), lr=args.scaleshift_lr)
    
    print("Initialized scale and shift.")
    ################################

    # create camera embedding function
    embedcam_fn = None

    # optimize nerf
    print('Begin')
    N_iters = args.num_iterations + 1
    global_step = start
    start = start + 1

    init_learning_rate = args.lrate
    init_learning_rate_cimle = args.cimle_lrate
    old_learning_rate = init_learning_rate
    old_learning_rate_cimle = init_learning_rate_cimle

    # if args.cimle_white_balancing and args.load_pretrained:
    if args.load_pretrained:
        if args.pretrained_dir is not None:
            path = args.pretrained_dir
            ckpts = [os.path.join(path, f) for f in sorted(os.listdir(path)) if '000.tar' in f]
        elif args.ft_path is not None:
            ckpts = [args.ft_path]
        print('Found ckpts', ckpts)
        ckpt_path = ckpts[-1]
        print('Reloading pretrained model from', ckpt_path)

        ckpt = torch.load(ckpt_path)

        coarse_model_dict = render_kwargs_train["network_fn"].state_dict()
        coarse_keys = {"module." + k.replace("module.", ""): v for k, v in ckpt['network_fn_state_dict'].items() if "module." + k.replace("module.", "") in coarse_model_dict} 

        fine_model_dict = render_kwargs_train["network_fine"].state_dict()
        fine_keys = {"module." + k.replace("module.", ""): v for k, v in ckpt['network_fine_state_dict'].items() if "module." + k.replace("module.", "") in fine_model_dict} 

        print("Num keys loaded:")
        print(len(coarse_keys.keys()), coarse_keys.keys())
        print(len(fine_keys.keys()), fine_keys.keys())

        
        coarse_model_dict.update(coarse_keys)
        fine_model_dict.update(fine_keys)
        render_kwargs_train["network_fn"].load_state_dict(coarse_model_dict)
        render_kwargs_train["network_fine"].load_state_dict(fine_model_dict)
        

        ## Load scale and shift
        DEPTH_SHIFTS = torch.load(ckpt_path)["depth_shifts"]
        DEPTH_SCALES = torch.load(ckpt_path)["depth_scales"] 
        
        if args.freeze_pretrained:
            for k, v in coarse_model_dict.items():
                v.requires_grad = False
            for k, v in fine_model_dict.items():
                v.requires_grad = False
            DEPTH_SHIFTS.requires_grad = False
            DEPTH_SCALES.requires_grad = False

        print("Scales:")
        print(DEPTH_SCALES)
        print()
        print("Shifts:")
        print(DEPTH_SHIFTS)

        print("Loaded depth shift/scale from pretrained model.")
        ########################################
        ########################################        
    model_fine = render_kwargs_train["network_fine"].module
    # init cache
    if model_fine is not None:
        model_fine.sample_random_fns()
        
    if args.do_log:
        render_kwargs_test["iteration"] = args.num_iterations
        mean_metrics_val, images_val = render_images_with_metrics(None, i_val, images, depths, valid_depths, \
            poses, H, W, intrinsics, lpips_alex, args, render_kwargs_test)
        tb.add_scalars('mse', {'val': mean_metrics_val.get("img_loss")}, 0)
        tb.add_scalars('psnr', {'val': mean_metrics_val.get("psnr")}, 0)
        tb.add_scalars('ssim', {'val': mean_metrics_val.get("ssim")}, 0)
        tb.add_scalars('lpips', {'val': mean_metrics_val.get("lpips")}, 0)

        if mean_metrics_val.has("depth_rmse"):
            tb.add_scalar('depth_rmse', mean_metrics_val.get("depth_rmse"), 0)
        if 'rgbs0' in images_val:
            tb.add_scalars('mse0', {'val': mean_metrics_val.get("img_loss0")}, 0)
            tb.add_scalars('psnr0', {'val': mean_metrics_val.get("psnr0")}, 0)
        val_image =  torch.cat((
            torchvision.utils.make_grid(images_val["rgbs"], nrow=1), \
            torchvision.utils.make_grid(images_val["target_rgbs"], nrow=1), \
            torchvision.utils.make_grid(images_val["depths"], nrow=1), \
            torchvision.utils.make_grid(images_val["target_depths"], nrow=1)), 2)
        tb.add_image('val_image', val_image , 0)
    
    
    for i in trange(start, N_iters):
        ### Scale the hypotheses by scale and shift
        img_i = np.random.choice(i_train)
        curr_scale = DEPTH_SCALES[img_i]
        curr_shift = DEPTH_SHIFTS[img_i]

        ## Scale and shift
        batch_rays, target_s, target_d, target_vd, img_i, target_h, space_carving_mask, curr_cached_u = get_ray_batch_from_one_image_hypothesis_idx(H, W, img_i, images, depths, valid_depths, poses, \
            intrinsics, all_depth_hypothesis, args, None, None)

        target_h = target_h*curr_scale + curr_shift        

        if args.input_ch_cam > 0:
            render_kwargs_train['embedded_cam'] = embedcam_fn[img_i]

        target_d = target_d.squeeze(-1)

        render_kwargs_train["cached_u"] = None
        render_kwargs_train["iteration"] = i
        render_kwargs_test["iteration"] = i
        # check whether to recache
        if i % args.i_cache == 0:
            if model_fine is not None:
                model_fine.sample_random_fns()

        rgb, _, _, extras = render_hyp(H, W, None, chunk=args.chunk, rays=batch_rays, verbose=i < 10, retraw=True,  is_joint=args.is_joint, **render_kwargs_train)

        # compute loss and optimize
        optimizer.zero_grad()
        optimizer_ss.zero_grad()
        optimizer_cimle.zero_grad()
        img_loss_total = (rgb - target_s) ** 2
        img_loss = img_loss_total.mean()  
        psnr = mse2psnr(img_loss)
        
        loss = img_loss

        if args.space_carving_weight>0. and i>args.warm_start_nerf:
            space_carving_loss = compute_space_carving_loss(extras["pred_hyp"], target_h, is_joint=args.is_joint, norm_p=args.norm_p, threshold=args.space_carving_threshold, mask=space_carving_mask)
            
            loss = loss + args.space_carving_weight * space_carving_loss
        else:
            space_carving_loss = torch.mean(torch.zeros([target_h.shape[0]]).to(target_h.device))
        if args.floater_M is not None:
            floater_loss = extras["floater_loss_coarse"].mean()
            if args.N_importance > 0:
                floater_loss = floater_loss + extras["floater_loss_fine"].mean()
            loss = loss + args.floater_weight * floater_loss
        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            psnr0 = mse2psnr(img_loss0)
            loss = loss + img_loss0
        # Direction loss 
        if args.direction_lambda > 0:
            gt_ds = F.normalize(batch_rays[1], dim=-1)
            pred_dirs = extras["directions"]
            global_z_val = torch.norm(batch_rays[1], dim=-1, p=2, keepdim=True) * extras["z_vals"]
            global_z_val = global_z_val.unsqueeze(-1).unsqueeze(-1)
            gt_ds = gt_ds.unsqueeze(-2).unsqueeze(-2)
            trans = extras["trans"].unsqueeze(-1).unsqueeze(-1)
            direction_loss = weighted_direction_loss(gt_ds, global_z_val, pred_dirs, trans, near, far, args.direction_loss_type, args.predict_z_val_type).mean()
            loss = loss + args.direction_lambda * direction_loss
        else:
            direction_loss = torch.zeros(1).to(device)
            
        if args.reg_lambda > 0:
            # TODO make threshold not arbitrary
            pred_dirs = extras["directions"]
            invisible_pts_mask = extras["trans"] < args.reg_trans_thresh
            # visible_pts_mask = extras["trans"] > 0.9
            visible_pts_start = (invisible_pts_mask.float().argmax(1) - 1).clip(min=0).reshape(-1, 1)
            visible_pts_end = torch.clip(visible_pts_start - args.point_take_per_ray, min=0).reshape(-1, 1)
            visible_pts_mask = torch.logical_and(torch.arange(invisible_pts_mask.shape[1]).reshape(1, -1).to(device) <= visible_pts_start, torch.arange(invisible_pts_mask.shape[1]).reshape(1, -1).to(device) > visible_pts_end)
            visible_pts_mask = torch.logical_and(visible_pts_mask, (invisible_pts_mask).any(1, keepdim=True))
            # import ipdb; ipdb.set_trace()
            # print(visible_pts_mask.shape)
            # TODO make threshold not a hard one 
            if args.direction_loss_type != -1:
                pred_trans = torch.norm(pred_dirs[..., :3], p=2, dim=-1)
                visible_direction_mask = pred_trans > args.reg_dirs_to_take_thresh
                # import ipdb; ipdb.set_trace()
                visible_direction_mask = torch.logical_and(visible_direction_mask, visible_pts_mask.unsqueeze(-1))
            else:
                visible_direction_mask = visible_pts_mask[..., None].repeat_interleave(args.cimle_sample_num, -1)
            
            if not visible_direction_mask.any():
                reg_loss = torch.zeros(1).to(device)
            else:
                valid_preds = pred_dirs[visible_direction_mask]
                rand_samples = torch.randperm(valid_preds.shape[0])[:args.N_rand]
                valid_preds = valid_preds[rand_samples]
                valid_pts = extras["pts"][..., None, :].repeat_interleave(args.cimle_sample_num, -2)[visible_direction_mask][rand_samples]
                valid_trans = extras["trans"][..., None].repeat_interleave(args.cimle_sample_num, -1)[visible_direction_mask][rand_samples]
                valid_preds.detach_()
                valid_pts.detach_()
                valid_trans.detach_()
                visible_dirs, visible_z_vals = torch.split(valid_preds, [3, 1], -1)
                visible_dirs = F.normalize(visible_dirs, dim=-1, p=2)
                if args.predict_z_val_type == 2:
                    visible_z_vals = visible_z_vals * (far - near) + near
                visible_os = -visible_dirs * visible_z_vals + valid_pts
                visible_sampled_ray_batches = torch.stack([visible_os, visible_dirs], 0)
                visible_sampled_ray_batches.detach_()
                #####  regualization optimization loop  #####
                _, _, _, pred_extras = render_hyp(H, W, None, chunk=args.chunk, rays=visible_sampled_ray_batches, verbose=i < 10, retraw=True,  is_joint=args.is_joint, **render_kwargs_train)
                # TODO figure out whether to match transmittance or weights
                if args.reg_loss_type == 0:
                    fine_index_mask = pred_extras["fine_index_mask"]
                    pred_trans = pred_extras["trans"][~fine_index_mask].reshape(visible_sampled_ray_batches.shape[1], args.N_samples)
                    pred_z_vals = pred_extras["z_vals"][~fine_index_mask].reshape(visible_sampled_ray_batches.shape[1], args.N_samples)
                elif args.reg_loss_type == -1:
                    pred_trans = pred_extras["trans"]
                    pred_z_vals = pred_extras["z_vals"]
                idu = torch.searchsorted(pred_z_vals.contiguous(), visible_z_vals.contiguous(), right=True)
                below = torch.max(torch.zeros_like(idu-1), idu-1)
                pred_trans = torch.gather(pred_trans, 1, below)
                # print(pred_trans[..., 0], valid_trans)
                # hinge loss
                trans_diff = valid_trans - pred_trans[..., 0]
                # print(trans_diff)
                # import ipdb; ipdb.set_trace()
                # print(trans_diff.max())
                reg_loss = F.relu(trans_diff - args.trans_margin).mean()
                # reg_loss = reg_loss[trans_diff > args.trans_margin].mean()
            

            loss = loss + args.reg_lambda * reg_loss
        else:
            reg_loss = torch.zeros(1).to(device)
        loss.backward()

        ### Update learning rate
        learning_rate = get_learning_rate(init_learning_rate, i, args.decay_step, args.decay_rate, staircase=True)
        if old_learning_rate != learning_rate:
            update_learning_rate(optimizer, learning_rate)
            old_learning_rate = learning_rate
        
        
        if not args.freeze_pretrained:
            optimizer.step()
        if i < args.freeze_ss and not args.freeze_pretrained:
            optimizer_ss.step()
        ### Update learning rate cIMLE
        learning_rate_cimle = get_learning_rate(init_learning_rate_cimle, i, args.decay_step, args.decay_rate, staircase=True)
        if old_learning_rate_cimle != learning_rate_cimle:
            update_learning_rate(optimizer, learning_rate_cimle)
            old_learning_rate_cimle = learning_rate_cimle
        if args.direction_lambda > 0:
            optimizer_cimle.step()
            
        

        ### Don't optimize scale shift for the last 100k epochs, check whether the appearance will crisp
        


        # write logs
        if i%args.i_weights==0:
            path = os.path.join(args.ckpt_dir, args.expname, '{:06d}.tar'.format(i))
            save_dict = {
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'optimizer_cimle_state_dict': optimizer_cimle.state_dict(),
            }
            if render_kwargs_train['network_fine'] is not None:
                save_dict['network_fine_state_dict'] = render_kwargs_train['network_fine'].state_dict()

            if args.input_ch_cam > 0:
                save_dict['embedded_cam'] = embedcam_fn

            save_dict['depth_shifts'] = DEPTH_SHIFTS
            save_dict['depth_scales'] = DEPTH_SCALES

            torch.save(save_dict, path)
            print('Saved checkpoints at', path)
        
        if i%args.i_print==0:
            s = f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}  MSE: {img_loss.item()} "

            if args.space_carving_weight > 0.:
                s += f"Space carving: {space_carving_loss.item()} "
            if args.direction_lambda > 0:
                s += f"Direction loss: {direction_loss.item()} "
            if args.reg_lambda > 0:
                s += f"Regularizer loss: {reg_loss.item()} "
            if args.floater_M is not None:
                s += f" Floater: {floater_loss.item()}"
            print(s)
            if args.do_log:
                tb.add_scalars('mse', {'train': img_loss.item()}, i)

                if args.space_carving_weight > 0.:
                    tb.add_scalars('space_carving_loss', {'train': space_carving_loss.item()}, i)
                if args.direction_lambda > 0:
                    tb.add_scalars('direction_loss', {'train': direction_loss.item()}, i)

                if args.reg_lambda > 0:
                    tb.add_scalars('reg_loss', {'train': reg_loss.item()}, i)
                
                tb.add_scalars('psnr', {'train': psnr.item()}, i)

        if i%args.i_img==0 and args.do_log:
            
            
            # compute validation metrics and visualize 8 validation images
            mean_metrics_val, images_val = render_images_with_metrics(None, i_val, images, depths, valid_depths, \
                poses, H, W, intrinsics, lpips_alex, args, render_kwargs_test)
            tb.add_scalars('mse', {'val': mean_metrics_val.get("img_loss")}, i)
            tb.add_scalars('psnr', {'val': mean_metrics_val.get("psnr")}, i)
            tb.add_scalars('ssim', {'val': mean_metrics_val.get("ssim")}, i)
            tb.add_scalars('lpips', {'val': mean_metrics_val.get("lpips")}, i)

            if mean_metrics_val.has("depth_rmse"):
                tb.add_scalar('depth_rmse', mean_metrics_val.get("depth_rmse"), i)
            val_image =  torch.cat((
                torchvision.utils.make_grid(images_val["rgbs"], nrow=1), \
                torchvision.utils.make_grid(images_val["target_rgbs"], nrow=1), \
                torchvision.utils.make_grid(images_val["depths"], nrow=1), \
                torchvision.utils.make_grid(images_val["target_depths"], nrow=1)), 2)
            tb.add_image('val_image', val_image , i)

        # test at the last iteration
        if (i + 1) == N_iters:
            torch.cuda.empty_cache()
            images = torch.Tensor(test_images).to(device)
            depths = torch.Tensor(test_depths).to(device)
            valid_depths = torch.Tensor(test_valid_depths).bool().to(device)
            poses = torch.Tensor(test_poses).to(device)
            intrinsics = torch.Tensor(test_intrinsics).to(device)
            mean_metrics_test, images_test = render_images_with_metrics(None, i_test, images, depths, valid_depths, \
                poses, H, W, intrinsics, lpips_alex, args, render_kwargs_test)
            write_images_with_metrics(images_test, mean_metrics_test, far, args)
            # tb.flush()

        global_step += 1

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('task', type=str, help='one out of: "train", "test", "test_with_opt", "video"')
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, default=None, 
                        help='specify the experiment, required for "test" and "video", optional for "train"')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=256,
                        help='batch size (number of random rays per gradient step)')


    ### Learning rate updates
    parser.add_argument('--num_iterations', type=int, default=500000, help='Number of epochs')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument('--decay_step', type=int, default=400000, help='Decay step for lr decay [default: 200000]')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='Decay rate for lr decay [default: 0.7]')


    parser.add_argument("--chunk", type=int, default=32*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk_per_gpu", type=int, default=32*64*2, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=128,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', default=True,
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=9,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=0,
                        help='log2 of max freq for positional encoding (2D direction)')


    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--lindisp", action='store_true', default=False,
                        help='sampling linearly in disparity rather than depth')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric logging')
    parser.add_argument("--i_img",     type=int, default=50000,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=50000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--ckpt_dir", type=str, default="",
                        help='checkpoint directory')

    # data options
    parser.add_argument("--scene_id", type=str, default="scene0758_00",
                        help='scene identifier')
    parser.add_argument("--data_dir", type=str, default="",
                        help='directory containing the scenes')

    ### Train json file --> experimenting making views sparser
    parser.add_argument("--train_jsonfile", type=str, default='transforms_train.json',
                        help='json file containing training images')

    parser.add_argument("--cimle_dir", type=str, default="dump_0826_pretrained_dd_scene0710_train/",
                        help='dump_dir name for prior depth hypotheses')
    parser.add_argument("--num_hypothesis", type=int, default=20, 
                        help='number of cimle hypothesis')
    parser.add_argument("--space_carving_weight", type=float, default=0.007,
                        help='weight of the depth loss, values <=0 do not apply depth loss')
    parser.add_argument("--warm_start_nerf", type=int, default=0, 
                        help='number of iterations to train only vanilla nerf without additional losses.')

    parser.add_argument('--scaleshift_lr', default= 0.0000001, type=float)
    parser.add_argument('--scale_init', default= 1.0, type=float)
    parser.add_argument('--shift_init', default= 0.0, type=float)
    parser.add_argument("--freeze_ss", type=int, default=400000, 
                            help='dont update scale/shift in the last few epochs')

    ### u sampling is joint or not
    parser.add_argument('--is_joint', default= False, type=bool)

    ### Norm for space carving loss
    parser.add_argument("--norm_p", type=int, default=2, help='norm for loss')
    parser.add_argument("--space_carving_threshold", type=float, default=0.0,
                        help='threshold to not penalize the space carving loss.')
    parser.add_argument('--mask_corners', default= False, type=bool)

    parser.add_argument('--load_pretrained', default= False, type=bool)
    parser.add_argument("--pretrained_dir", type=str, default=None,
                        help='folder directory name for where the pretrained model that we want to load is')

    parser.add_argument("--input_ch_cam", type=int, default=0,
                        help='number of channels for camera index embedding')

    parser.add_argument("--opt_ch_cam", action='store_true', default=False,
                        help='optimize camera embedding')    
    parser.add_argument('--ch_cam_lr', default= 0.0001, type=float)
    
    
    ### cIMLE direction flags 
    parser.add_argument("--direction_lambda", type=float, default=1, help='direction loss weight')
    parser.add_argument("--predict_z_val_type", type=int, default=2, help='type of z val prediction')
    parser.add_argument("--direction_loss_type", type=int, default=0, help='type of direction loss')
    parser.add_argument("--cimle_lrate", type=float, default=5e-4, help='cIMLE learning rate')
    parser.add_argument("--cimle_latent_dim", type=int, default=32, help='cIMLE latent dim')
    parser.add_argument("--cimle_sample_num", type=int, default=16, help='cIMLE sample num')
    parser.add_argument("--random_fn_num_layers", type=int, default=1, help='number of layers to use for random fns')
    parser.add_argument("--direction_num_layers", type=int, default=3, help='number of layers for direction branch')
    parser.add_argument("--use_bias", type=bool, default=True, help='whether to use bias for random fns')
    parser.add_argument("--do_log", action='store_true', help='whether to log the training process')
    parser.add_argument("--freeze_pretrained", action='store_true', help='Set to true if to freeze pretrained model')
    parser.add_argument("--i_cache", type=int, default=1000, help='cache interval')
    
    
    
    ### NeRF regularization
    parser.add_argument("--reg_lambda", type=float, default=0, help='regularizer loss weight')
    parser.add_argument("--reg_trans_thresh", type=float, default=0.9, help='transmittance thresh for regularizer')
    parser.add_argument("--reg_dirs_to_take_thresh", type=float, default=0.9, help='transmittance thresh for regularizer')
    parser.add_argument("--reg_loss_type", type=int, default=-1, help='transmittance thresh for regularizer')
    parser.add_argument("--trans_margin", type=float, default=0.05, help='margin for the hinge loss')
    parser.add_argument('--restart', default= False, type=bool)
    parser.add_argument("--ft_path", type=str, default=None,
                        help='tar path for where the pretrained model that we want to load is')
    parser.add_argument("--point_take_per_ray", type=int, default=10, help='cache interval')
    parser.add_argument("--thresh_by_depth_hyp", type=bool, default=False, help='whether to threshold by depth hypothesis')
    parser.add_argument('--use_pseudo_cams', default= False, type=bool)
    parser.add_argument("--warm_start_pseudo_cam_iter", type=int, default=500)
    parser.add_argument("--num_iter_for_dir_loss", type=int, default=-1)
    
    parser.add_argument('--reg_percent', default= 0, type=float)
    parser.add_argument('--floater_weight', default= 0.01, type=float)
    parser.add_argument('--floater_M', default=None, type=int)
    ##################################

    return parser

def run_nerf():
    
    parser = config_parser()
    args = parser.parse_args()


    if args.task == "train":
        if args.expname is None:
            args.expname = "{}_{}".format(datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S'), args.scene_id)
        args_file = os.path.join(args.ckpt_dir, args.expname, 'args.json')
    
    os.makedirs(os.path.join(args.ckpt_dir, args.expname), exist_ok=True)
    if args.task == "train":
        with open(args_file, 'w') as af:
            json.dump(vars(args), af, indent=4)

    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))

    # Multi-GPU
    args.n_gpus = torch.cuda.device_count()
    print(f"Using {args.n_gpus} GPU(s).")

    # Load data
    scene_data_dir = os.path.join(args.data_dir, args.scene_id)

    images, depths, valid_depths, poses, H, W, intrinsics, near, far, i_split, \
    gt_depths, gt_valid_depths, all_depth_hypothesis = load_scene_scannet(scene_data_dir, args.cimle_dir, args.num_hypothesis, 'transforms_train.json')

    i_train, i_val, i_test, i_video = i_split

    # Compute boundaries of 3D space
    max_xyz = torch.full((3,), -1e6)
    min_xyz = torch.full((3,), 1e6)
    for idx_train in i_train:
        rays_o, rays_d = get_rays(H, W, torch.Tensor(intrinsics[idx_train]), torch.Tensor(poses[idx_train])) # (H, W, 3), (H, W, 3)
        points_3D = rays_o + rays_d * far # [H, W, 3]
        max_xyz = torch.max(points_3D.view(-1, 3).amax(0), max_xyz)
        min_xyz = torch.min(points_3D.view(-1, 3).amin(0), min_xyz)
    args.bb_center = (max_xyz + min_xyz) / 2.
    args.bb_scale = 2. / (max_xyz - min_xyz).max()
    print(args.bb_center, args.bb_scale)
    print("Computed scene boundaries: min {}, max {}".format(min_xyz, max_xyz))

    scene_sample_params = {
        'precomputed_z_samples' : None,
        'near' : near,
        'far' : far,
    }

    lpips_alex = LPIPS().to(device)

    if args.task == "train":
        train_nerf(images, depths, valid_depths, poses, intrinsics, i_split, args, scene_sample_params, lpips_alex, gt_depths, gt_valid_depths, all_depth_hypothesis)
    
 
    # create nerf model for testing
    ckpts = []
    if args.ft_path is not None:
        print('Reloading from ft_path', args.ft_path)
        ckpts += [args.ft_path]
    else:
        ckpts = load_checkpoint(args, ret_all=True)
        ckpts = [ckpts[-1]]
    count = len(ckpts)
    for n, ckpt in enumerate(ckpts):
        print(f"Testing ckpt: {ckpt}")

        torch.cuda.empty_cache()
        _, render_kwargs_test, step, nerf_grad_vars, _, nerf_grad_names, cimle_nerf_grad_vars, _, cimle_nerf_grad_names = create_nerf(args, scene_sample_params, ft_path=ckpt)
        render_kwargs_test["iteration"] = args.num_iterations
        for param in nerf_grad_vars + cimle_nerf_grad_vars:
            param.requires_grad = False

        # render test set and compute statistics
        with_test_time_optimization = False

        if args.task == "video":
            if n != count - 1:
                continue
            poses = torch.Tensor(poses[i_video]).to(device)
            intrinsics = torch.Tensor(intrinsics[i_video]).to(device)
            render_video(poses, H, W, intrinsics, str(args.num_iterations), args, render_kwargs_test)
            # i_video = i_video - i_video[0]
            # count = len(i_video)
            # rgbs_res = []
            # depths_res = []
            # for i in trange(0, len(poses)):
            #     pose = poses[i, :3, :4]
            #     intrinsic = intrinsics[i]
            #     print("Render image {}/{}".format(i + 1, count), end="")

            #     render_kwargs_test["embedded_cam"] = embedcam_fn[img_idx]
        
            #     with torch.no_grad():
            #         rgb, _, _, extras = render(H, W, intrinsic, chunk=args.N_rand, c2w=pose, ret_dirs=False, **render_kwargs_test)
                    
            #         rgb = torch.clamp(rgb, 0, 1)
                    
            #         # store result
            #         rgbs_res.append(rgb.clamp(0., 1.).permute(2, 0, 1).cpu().numpy())
            #         depths_res.append((extras['depth_map'] / far).unsqueeze(0).cpu().numpy())
            # moviebase = os.path.join(basedir, expname, '{}_video_{:06d}_'.format(expname, i))
            # imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            # imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)
            print("Done")
            return


        if args.task == "test_opt":
            with_test_time_optimization = True
        images = torch.Tensor(images[i_test]).to(device)
        if gt_depths is None:
            depths = torch.Tensor(depths[i_test]).to(device)
            valid_depths = torch.Tensor(valid_depths[i_test]).bool().to(device)
        else:
            depths = torch.Tensor(gt_depths[i_test]).to(device)
            valid_depths = torch.Tensor(gt_valid_depths[i_test]).bool().to(device)
        poses = torch.Tensor(poses[i_test]).to(device)
        intrinsics = torch.Tensor(intrinsics[i_test]).to(device)
        i_test = i_test - i_test[0]
        mean_metrics_test, images_test = render_images_with_metrics(None, i_test, images, depths, valid_depths, poses, H, W, intrinsics, lpips_alex, args, \
            render_kwargs_test, with_test_time_optimization=with_test_time_optimization)
        write_images_with_metrics(images_test, mean_metrics_test, far, args, with_test_time_optimization=with_test_time_optimization, step=step+1)

if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    run_nerf()
