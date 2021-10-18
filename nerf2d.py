import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision.transforms import Normalize
from torchvision.models import vgg16
import numpy as np
from itertools import chain
import os

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

class Nerf2d_slot(nn.Module):
    def __init__(self, position_emb_dim=5, mlp_hidden_size=64):
        super().__init__()
        input_dim = position_emb_dim*2*2 +2
        self.mlp_hidden_size = mlp_hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_size), 
            nn.ReLU(True),
            nn.Linear(mlp_hidden_size, mlp_hidden_size), 
            nn.ReLU(True),
            nn.Linear(mlp_hidden_size, mlp_hidden_size), 
            nn.ReLU(True),
            nn.Linear(mlp_hidden_size, mlp_hidden_size), 
            nn.ReLU(True),
            nn.Linear(mlp_hidden_size, mlp_hidden_size//4), 
            nn.ReLU(True),
            nn.Linear(mlp_hidden_size//4, 4))

    def forward(self, x):
        # x: (P, 2)
        encoded_pts = embedding_fn(x)

        rgba = self.mlp(encoded_pts) # (P, 4)

        rgb = (rgba[:, :3].tanh() + 1) / 2
        alpha = F.relu(rgba[:, -1:])
        rgba = torch.cat([rgb, alpha], dim=-1)

        return rgba


class Nerf2d():
    def __init__(self, image, N_nerf=3, lr=5e-4, device='cuda:0'):
        super().__init__()
        self.image = image
        self.N_nerf = N_nerf
        self.NeRFs = []
        self.device = device
        for _ in range(N_nerf):
            self.NeRFs.append(Nerf2d_slot().to(device))
        # self.norm = nn.BatchNorm1d(N_nerf).to(device)

        self.optimizer = optim.Adam(chain(
                *[nerf.parameters() for nerf in self.NeRFs]), lr=lr)
    

    def update(self, val=False, print_=False, iter=-1):
        H, W, C = self.image.shape
        pts = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W)), 
                    dim=-1).reshape(-1, 2)
        pts = pts.to(self.device)
        rgbas = []
        for k in range(self.N_nerf):
            rgba_slot = self.NeRFs[k](pts)
            rgbas.append(rgba_slot)
        rgbas = torch.stack(rgbas, dim=0) # (K, N, 4)

        rgbs = rgbas[..., 0:3]
        alphas = rgbas[..., -1]
        
        eps = 1e-5
        # alpha_norm = (alphas - torch.mean(alphas, dim=1, keepdim=True) + 1) \
        #         / (torch.var(alphas, dim=1, keepdim=True) + eps)
        # alpha_norm = F.relu(alpha_norm)
        weight_slot = (alphas / (torch.sum(alphas, dim=0) + eps))[..., None] # (K, N)
        # weight_slot_norm = weight_slot.unsqueeze(0) # (1, K, N)
        # weight_slot_norm = self.norm(weight_slot_norm)
        # weight_slot_norm = weight_slot_norm.squeeze() # (K, N)
        # weight_slot = weight_slot[..., None]
        # mask = torch.argmax(weight_slot_norm, dim=0) # (N, 1)
        # rgb = rgbs[mask, torch.arange(mask.shape[0]), :]
        rgb = torch.sum(weight_slot * rgbs, dim=0)

        loss_recon = F.mse_loss(rgb, self.image.reshape([-1, 3]))

        # overlap loss
        max_alpha, _ = torch.max(alphas, dim=0) # (N, 1)
        sum_alpha = torch.sum(alphas, dim=0)
        loss_overlap = (sum_alpha - max_alpha).mean()

        # loss sim: expectation
        
        weight_pixel = weight_slot / (torch.sum(weight_slot, dim=1, keepdim=True) + eps) # (K, N, 1)
        mu = torch.sum(weight_pixel * rgbs, dim=1, keepdim=True) # (K, 1, 3)
        var = torch.sum(((rgbs - mu) ** 2) * weight_pixel, dim=1, keepdim=True) # (K, 1, 3)


        loss_sim = var.sum()

        # regu
        loss_regu = torch.sum((torch.mean(weight_slot, dim=1) - 1/3) ** 2)

        loss = loss_recon + 0.5 * loss_overlap + 0.01 * loss_sim + loss_regu


        # update 
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if print_:
            print('='*50)
            print(iter)
            print('loss:         ', loss)
            print('loss_recon:   ', loss_recon)
            print('loss_overlap: ', loss_overlap)
            print('loss sim:     ', loss_sim)
            print('var:          ', var)
            print('loss regu:    ', loss_regu)
            print('='*50)

        if val:
            rgbs *= weight_slot
            return rgb.reshape((H, W, -1)), rgbs.reshape((self.N_nerf, H, W, -1))

        return loss, loss_recon, loss_overlap, loss_sim


