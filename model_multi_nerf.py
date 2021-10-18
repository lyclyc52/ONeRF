import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision.transforms import Normalize
from torchvision.models import vgg16
import numpy as np
from itertools import chain
import os

def get_rays(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    device = c2w.device

    i, j = torch.meshgrid(torch.arange(W),
                       torch.arange(H))
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    dirs = dirs.to(device)
    dirs = dirs[None,...]
    N = c2w.shape[0]
    dirs = dirs.expand([N, -1, -1, -1])

    rays_d = torch.sum(dirs[..., None, :] * c2w[:, None, None, :3, :3], -1)


    rays_o = c2w[:,:3, -1]
    rays_o = rays_o[:,None, None, :]


    rays_o = rays_o.expand([-1,H,W,-1])

    return rays_o, rays_d

def sampling_points(cam_param, c2w, N_samples=64, near=4., far=14., 
        is_selection=False, N_selection=64*32):
    H, W, focal = cam_param
    batch_size = c2w.shape[0]
    device = c2w.device
    rays_o, rays_d = get_rays(H, W, focal, c2w) 

    t_vals = torch.linspace(0., 1., N_samples)
    t_vals = t_vals.to(device)

    z_vals = near * (1.-t_vals) + far * (t_vals)
    z_vals = z_vals[None, None, None, :]

    z_vals = z_vals.expand([batch_size, H, W, -1])


    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], -1)
    lower = torch.cat([z_vals[..., :1], mids], -1)
    # stratified samples in those intervals
    t_rand = torch.rand(z_vals.shape).to(device)
    z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., None]  # [N_rays, N_samples, 3]


    if is_selection:
        # coords = torch.stack(torch.meshgrid(
        #         torch.range(H), torch.range(W), indexing='ij'), -1)
        # coords = torch.reshape(coords, [-1, 2])
        # select_inds = np.random.choice(
        #     coords.shape[0], size=[N_selection], replace=False)
        # # pts = torch.reshape(pts,[batch_size, -1, N_samples, 3])
        # # pts = torch.reshape(pts,[batch_size, -1, N_samples])


        # select_inds = torch.gather_nd(coords, select_inds[:, torch.newaxis])
        # # select_inds = torch.tile(select_inds[torch.newaxis,...],[batch_size,1,1])
        # select_inds = select_inds[torch.newaxis,...]


        # pts = torch.gather_nd(pts, select_inds, batch_dims = 1)
        # z_vals = torch.gather_nd(z_vals, select_inds, batch_dims = 1)
        # rays_d = torch.gather_nd(rays_d, select_inds, batch_dims = 1)

        pts = torch.reshape(pts,[batch_size, -1, N_samples, 3])
        z_vals = torch.reshape(z_vals,[batch_size, -1, N_samples])
        rays_d = torch.reshape(rays_d,[batch_size, -1, 3])


        select_inds = np.random.choice(pts.shape[1], size=N_selection, replace=False)

        pts = pts[:,select_inds,...]
        z_vals = z_vals[:,select_inds,...]
        rays_d = rays_d[:,select_inds,...]

        return pts, z_vals, rays_d, select_inds

    pts = torch.reshape(pts,[batch_size, -1, N_samples, 3])
    z_vals = torch.reshape(z_vals,[batch_size, -1, N_samples])
    rays_d = torch.reshape(rays_d,[batch_size, -1, 3])
    return pts, z_vals, rays_d

def embedding_fn(x, n_freq=5, keep_ori=True):
    """
    create sin embedding for 3d coordinates
    input:
        x: Px3
        n_freq: number of raised frequency
    """
    embedded = []
    if keep_ori:
        embedded.append(x)
    emb_fns = [torch.sin, torch.cos]
    freqs = 2. ** torch.linspace(0., n_freq - 1, steps=n_freq)
    for freq in freqs:
        for emb_fn in emb_fns:
            embedded.append(emb_fn(freq * x))
    embedded_ = torch.cat(embedded, dim=1)
    return embedded_

def raw2outputs(raw, z_vals, rays_d):   
    raw2alpha = lambda x, y: 1. - torch.exp(-x * y)
    device = raw.device

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.tensor([1e-2], device=device).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)
    rgb = raw[..., :3]

    alpha = raw2alpha(raw[..., 3], dists)  # [N_rays, N_samples]

    weights = alpha * torch.cumprod(torch.cat([torch.ones_like(alpha[...,0:1], device=device), 1. - alpha + 1e-10], -1), -1)[...,:-1]

    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    weights_norm = weights.detach() + 1e-5
    weights_norm /= weights_norm.sum(dim=-1, keepdim=True)
    depth_map = torch.sum(weights_norm * z_vals, -1)

    rgb_map = rgb_map.permute([1, 0])

    return rgb_map


class Encoder_VGG(nn.Module):
    def __init__(self, input_c=3, layers_c=64):

        super().__init__()

        vgg_features = vgg16(pretrained=True).features
        self.feature_extractor = nn.Sequential()
        for i in range(16):
            self.feature_extractor.add_module(str(i), vgg_features[i])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.vgg_preprocess = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.upsample = nn.Upsample(scale_factor=4)
        self.input_c=input_c
        self.layers_c=layers_c

    def forward(self, x):
        """
        input:
            x: input image, BxHxWx3
        output:
            feature_map: BxCxHxW
        """
        x = self.vgg_preprocess(x)
        x = self.feature_extractor(x)
        x = self.upsample(x)

        return x

class NeRF_Ind(nn.Module):
    def __init__(self, position_emb_dim=5, mlp_hidden_size=64):
        super().__init__()

        input_dim = position_emb_dim*3*2 +3
        self.mlp_hidden_size = mlp_hidden_size

        self.decoder_mlp_layer_0 = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_size), 
            nn.ReLU(True),
            nn.Linear(mlp_hidden_size, mlp_hidden_size), 
            nn.ReLU(True),
            nn.Linear(mlp_hidden_size, mlp_hidden_size), 
            nn.ReLU(True))

        self.decoder_mlp_layer_1 =  nn.Sequential(        
            nn.Linear(mlp_hidden_size+input_dim, mlp_hidden_size), 
            nn.ReLU(True),
            nn.Linear(mlp_hidden_size, mlp_hidden_size), 
            nn.ReLU(True),
            nn.Linear(mlp_hidden_size, mlp_hidden_size), 
            nn.ReLU(True))

        self.decoder_density = nn.Linear(mlp_hidden_size, 1)

        self.decoder_latent = nn.Linear(mlp_hidden_size, mlp_hidden_size)
        self.decoder_color = nn.Sequential(nn.Linear(mlp_hidden_size, mlp_hidden_size//4),
                                     nn.ReLU(True),
                                     nn.Linear(mlp_hidden_size//4, 3))


    def forward(self, pts):
        # pts: (P*D, 3)

        encoded_pts = embedding_fn(pts)

        chunk = 1024*32
        output = []
        input = encoded_pts

        for i in range(0, input.shape[0], chunk):
            tmp = self.decoder_mlp_layer_0(input[i:i+chunk])
            tmp = self.decoder_mlp_layer_1(torch.cat([input[i:i+chunk], tmp], dim=-1))

            raw_density = self.decoder_density(tmp)
            latent = self.decoder_latent(tmp)  
            raw_color = self.decoder_color(latent)  
             
            output_c = torch.cat([raw_color,raw_density], dim=-1)

            output.append(output_c)

        all_raws = torch.cat(output, dim=0)
        # activate the output
        raw_densities = F.relu(all_raws[:, -1:], True)
        raw_rgbs = (all_raws[:, :3].tanh() + 1) / 2
        all_raws = torch.cat([raw_densities, raw_rgbs], dim=-1)

        return all_raws


class Multiple_NeRF():
    def __init__(self, images, poses, hwf, N_nerf=2, lr=5e-4,
            device='cuda:0', masks=None):
        super().__init__()
        self.images = images.to(device)
        self.poses = poses.to(device)
        self.hwf = hwf
        self.masks = None
        if masks is not None:
            self.masks = masks # in same sequence
        feature_extractor = Encoder_VGG().to(device)
        self.features = feature_extractor(images)
        self.NeRFs = []
        self.N_nerf = N_nerf
        for _ in range(N_nerf):
            self.NeRFs.append(NeRF_Ind().to(device))

        # train tools
        self.optimizer = optim.Adam(chain(
                *[nerf.parameters() for nerf in self.NeRFs]), lr=lr)
        

    def update(self, samples_per_ray=64):
        
        # sample
        N = self.images.shape[0]
        img_id = np.random.randint(0, N, 1) # todo: multi-imgs?

        c2w = self.poses[img_id]
        pts, z_vals, rays_d, select_inds = sampling_points(self.hwf, 
                c2w, N_samples=samples_per_ray, near=4.0, far=14.0, 
                is_selection=True, N_selection=1024)  # (B, S, D, 3)
        pts = torch.reshape(pts, [-1, 3]) # (N, 3)
        # network
        raws = []
        for k in range(self.N_nerf):
            raws_local = self.NeRFs[k](pts)
            raws.append(raws_local)
        raws = torch.stack(raws, dim=0) # (K, N, 4)
        raw_densities = raws[..., -1:]
        masks = raw_densities / (raw_densities.sum(dim=0) + 1e-5)  # KxNx1
        
        masked_raws = masks * raws[..., :3]
        masked_raws = torch.cat([masked_raws, raw_densities], dim=-1) # todo: check
        combined_raws = masked_raws.sum(dim=0)


        # loss
        combined_raws = combined_raws.reshape([-1,samples_per_ray,4])
        z_vals = z_vals.reshape([-1,samples_per_ray])
        rays_d = rays_d.reshape([-1,3])
        rgb = raw2outputs(combined_raws, z_vals, rays_d)

        img = self.images[img_id].reshape([3, -1])
        img = img[:, select_inds]
        loss_recon = F.mse_loss(rgb, img)
        # loss_recon = 0

        if self.masks is not None:
            mask = self.masks[img_id].reshape((self.N_nerf, -1, 1))
            img = self.images[img_id].reshape([3, -1]).permute([1, 0])
            mask = mask[:, select_inds, :]
            img = img[select_inds]
            raw1 = raws[0:1].reshape([-1, samples_per_ray, 4])
            raw2 = raws[1:2].reshape([-1, samples_per_ray, 4])
            raw3 = raws[2:3].reshape([-1, samples_per_ray, 4])
            # raws = raws.reshape([self.N_nerf, -1, samples_per_ray, 4])
            rgb1 = raw2outputs(raw1, z_vals, rays_d).permute([1, 0])
            rgb2 = raw2outputs(raw2, z_vals, rays_d).permute([1, 0])
            rgb3 = raw2outputs(raw3, z_vals, rays_d).permute([1, 0])
            loss1 = F.mse_loss(rgb1*mask[0], img*mask[0])
            loss2 = F.mse_loss(rgb2*mask[1], img*mask[1])
            loss3 = F.mse_loss(rgb3*mask[2], img*mask[2])

            loss_recon += (loss1 + loss2 + loss3)

            

        # overlap loss
        max_sigma, _ = torch.max(raw_densities, dim=0) # (N, 1)

        combined_sigma = combined_raws.reshape([-1, 4])[..., -1:]
        loss_overlap = (combined_sigma - max_sigma).mean()

        loss = loss_recon #+ loss_overlap

        # update 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.masks is not None:
            return loss, loss_recon, loss_overlap, loss1, loss2
        else:
            return loss, loss_recon, loss_overlap


    def validate(self, samples_per_ray=64):
        with torch.no_grad():
            N, _, H, W = self.images.shape
            img_id = np.random.randint(0, N, 1)
            c2w = self.poses[img_id]
            pts, z_vals, rays_d = sampling_points(self.hwf, 
                    c2w, N_samples=samples_per_ray, near=4.0, far=14.0, 
                    is_selection=False)  # (B, S, D, 3)
            pts = torch.reshape(pts, [-1, 3]) # (N, 3)
            # network
            raws = []
            for k in range(self.N_nerf):
                raws_local = self.NeRFs[k](pts)
                raws.append(raws_local)
            raws = torch.stack(raws, dim=0) # (K, N, 4)
            raw_densities = raws[..., -1:]
            masks = raw_densities / (raw_densities.sum(dim=0) + 1e-5)  # KxNx1

            masked_raws = masks * raws
            combined_raws = masked_raws.sum(dim=0)

            combined_raws = combined_raws.reshape([-1,samples_per_ray,4])
            z_vals = z_vals.reshape([-1,samples_per_ray])
            rays_d = rays_d.reshape([-1,3])
            rgb = raw2outputs(combined_raws, z_vals, rays_d)
            rgb = rgb.reshape((3, H, W)).permute([1, 2, 0])

            rgb_locals = []
            if self.masks is not None:
                mask = self.masks[img_id]
            for k in range(self.N_nerf): # todel
                rgb_local = raw2outputs(raws[k].reshape([-1,samples_per_ray,4]), 
                    z_vals, rays_d)
                rgb_local = rgb_local.reshape((3, H, W)).permute([1, 2, 0])
                rgb_local = rgb_local * mask[0, k,..., None]
                rgb_locals.append(rgb_local)
            rgb = torch.sum(torch.stack(rgb_locals, dim=0), dim=0)

        return rgb, rgb_locals

    def save_weights(self, path, i):
        nerf_state_dicts = []
        for k in range(self.N_nerf):
            nerf_state_dicts.append(self.NeRFs[k].state_dict())
        torch.save({
            'epoch': i,
            'decoder_state_dict': nerf_state_dicts,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
