import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn import init
from torchvision.models import vgg16
from torch import autograd
import numpy as np
import time

from itertools import chain

from torchvision.transforms import Normalize
import os

def get_position(ray_o, ray_d, depth):
    '''
    input: ray origin, ray direction, depth
    output: 3d position of each points
    '''

    depth = depth[...,None]


    position = depth*ray_d+ray_o
    return position

def get_rays(H, W, focal, c2w,
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),):
    """Get ray origins, directions from a pinhole camera."""
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



def sampling_points(cam_param, c2w, N_samples=64, near=4., far=14., is_selection=False, N_selection=64*32):
    H, W, focal = cam_param
    batch_size = c2w.shape[0]
    device = c2w.device
    rays_o, rays_d = get_rays(H, W, focal, c2w, device) 

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


def preprocess_pts(points, num_slots):
    points_bg = points
    points_fg = points[None,...]
    points_fg = points_fg.expand([num_slots-1, -1, -1, -1, -1])

    points_bg = torch.reshape(points_bg, [-1,3])
    points_fg = torch.reshape(points_fg, [num_slots-1, -1, 3])

    return points_bg, points_fg






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

    return rgb_map

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


def L2_loss(x,y):
    loss = torch.mean((x-y)**2)
    return loss


class Encoder(nn.Module):
    def __init__(self, cam_param, device, input_c=3, layers_c=64, position_emb=False, position_emb_dim=3):

        super().__init__()
        self.device = device
        self.input_c=input_c
        self.layers_c=layers_c
        self.position_emb=position_emb
        self.position_emb_dim=position_emb_dim


        self.H, self.W, self.focal=cam_param
        if self.position_emb:
            self.input_c = self.input_c + position_emb_dim

        self.enc_down_0 = nn.Sequential(nn.Conv2d(self.input_c, layers_c, 5, stride=1, padding=2),
                                        nn.ReLU(True))
        self.enc_down_1 = nn.Sequential(nn.Conv2d(layers_c, layers_c, 5, stride=1, padding=2),
                                        nn.ReLU(True))
        self.enc_down_2 = nn.Sequential(nn.Conv2d(layers_c, layers_c, 5, stride=1, padding=2),
                                        nn.ReLU(True))
        self.enc_down_3 = nn.Sequential(nn.Conv2d(layers_c, layers_c, 5, stride=1, padding=2),
                                        nn.ReLU(True))
        self.enc_down_4 = nn.Sequential(nn.Conv2d(layers_c, layers_c, 5, stride=1, padding=2),
                                        nn.ReLU(True))

    def forward(self, x, depth, c2ws, is_selection=False):
        """
        input:
            x: input image, Bx3xHxW
        output:
            feature_map: BxCxHxW
        """


        if self.position_emb:
            rays_o, rays_d = get_rays(self.H, self.W, self.focal, c2ws) 
            position = get_position(rays_o, rays_d, depth)
            x = torch.cat([x, position], dim=-1)


        x = x.permute([0,3,1,2])

        x_down_0 = self.enc_down_0(x)
        x_down_1 = self.enc_down_1(x_down_0)

        x_down_2 = self.enc_down_2(x_down_1)
        x_down_3 = self.enc_down_3(x_down_2)
        x_down_4 = self.enc_down_3(x_down_3)

        return x_down_4

class Encoder_VGG(nn.Module):
    def __init__(self, cam_param, 
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), 
        input_c=3, layers_c=64, position_emb=True, position_emb_dim=3):

        super().__init__()

        self.device = device

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
        self.position_emb=position_emb
        self.position_emb_dim=position_emb_dim


        self.H, self.W, self.focal=cam_param
        if self.position_emb:
            self.input_c = self.input_c + position_emb_dim

    def forward(self, x, depth, c2ws, is_selection=False):
        """
        input:
            x: input image, BxHxWx3
        output:
            feature_map: BxCxHxW
        """
        x = x.permute([0,3,1,2])
        x = self.vgg_preprocess(x)
        x = self.feature_extractor(x)
        x = self.upsample(x)
        x = x.permute([0,2,3,1])

        if self.position_emb:
            rays_o, rays_d = get_rays(self.H, self.W, self.focal, c2ws, device=self.device) 
            position = get_position(rays_o, rays_d, depth)
            x = torch.cat([x, position], dim=-1)

        x = x.permute([0,3,1,2])

        return x

class SlotAttention(nn.Module):
    def __init__(self, 
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            num_iterations=2, num_slots=3, in_dim=64, slot_dim=64, mlp_hidden_size=128, epsilon=1e-8):
        super().__init__()

        self.device = device
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.in_dim =in_dim
        self.slot_dim = slot_dim
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon

        self.norm_inputs = nn.LayerNorm(in_dim)

        self.slots_mu_fg = nn.Parameter(torch.randn(1, slot_dim))
        self.slots_logsigma_fg = nn.Parameter(torch.zeros(1, slot_dim))
        init.xavier_uniform_(self.slots_logsigma_fg)
        self.slots_mu_bg = nn.Parameter(torch.randn(1, slot_dim))
        self.slots_logsigma_bg = nn.Parameter(torch.zeros(1, slot_dim))
        init.xavier_uniform_(self.slots_logsigma_bg)


        self.to_k = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_q_fg = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))
        self.to_q_bg = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))

        self.gru_fg = nn.GRUCell(slot_dim, slot_dim)
        self.gru_bg = nn.GRUCell(slot_dim, slot_dim)


        self.mlp_fg = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, slot_dim)
        )
        self.mlp_bg = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, slot_dim)
        )



    def forward(self, x):
        """
        input:
            feat: visual feature with position information, BxNxC
        output: slots: KxC, attn: KxN
        """



        mu_fg = self.slots_mu_fg.expand(self.num_slots-1, -1)
        sigma_fg = self.slots_logsigma_fg.exp().expand(self.num_slots-1, -1)
        slot_fg = mu_fg + sigma_fg * torch.randn_like(mu_fg)
        mu_bg = self.slots_mu_bg.expand(1, -1)
        sigma_bg = self.slots_logsigma_bg.exp().expand(1, -1)
        slot_bg = mu_bg + sigma_bg * torch.randn_like(mu_bg)

        x = self.norm_inputs(x)

        k_input = self.to_k(x)
        v_input = self.to_v(x)

        for _ in range(self.num_iterations):
            slot_prev_bg = slot_bg
            slot_prev_fg = slot_fg

            q_slots_fg = self.to_q_fg(slot_fg)
            q_slots_fg = q_slots_fg*self.slot_dim ** -0.5
            q_slots_bg = self.to_q_bg(slot_bg)
            q_slots_bg = q_slots_bg*self.slot_dim ** -0.5

            attn_logits_fg = torch.matmul(k_input, q_slots_fg.T)
            attn_logits_bg = torch.matmul(k_input, q_slots_bg.T)
            attn_logits = torch.cat([attn_logits_bg, attn_logits_fg], dim=-1)
            
            attn = attn_logits.softmax(dim=-1)

            attn_bg, attn_fg = attn[:, 0:1], attn[:, 1:]  # Nx1, Nx(K-1)
            weight_bg = attn_bg + self.epsilon
            weight_bg = weight_bg / torch.sum(weight_bg, dim=-2, keepdims=True)
            updates_bg = torch.matmul(weight_bg.T, v_input)

            weight_fg = attn_fg + self.epsilon
            weight_fg = weight_fg / torch.sum(weight_fg, dim=-2, keepdims=True)
            updates_fg = torch.matmul(weight_fg.T, v_input)



            slot_bg = self.gru_bg(
                updates_bg,
                slot_prev_bg
            )

            slot_fg = self.gru_fg(
                updates_fg.reshape(-1, self.slot_dim),
                slot_prev_fg.reshape(-1, self.slot_dim)
            )

            slot_bg = slot_bg + self.mlp_bg(slot_bg)
            slot_fg = slot_fg + self.mlp_fg(slot_fg)

        slots = torch.cat([slot_bg, slot_fg], dim=0)
        return slots, attn.T



class SlotAttention_Three(nn.Module):
    def __init__(self, device, num_iterations=2, num_slots=3, in_dim=64, slot_dim=64, mlp_hidden_size=128, epsilon=1e-8):
        super().__init__()

        self.device = device
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.in_dim =in_dim
        self.slot_dim = slot_dim
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon

        self.norm_inputs = nn.LayerNorm(in_dim)
        self.to_k = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(in_dim, slot_dim, bias=False)
        

        self.slots_mu = nn.ParameterList()
        self.slots_logsigma = nn.ParameterList()
        self.to_q = nn.ModuleList()
        self.gru = nn.ModuleList()
        self.mlp = nn.ModuleList()
        for i in range(num_slots):
            self.slots_mu.append(nn.Parameter(torch.randn(1, slot_dim)))
            self.slots_logsigma.append(nn.Parameter(torch.zeros(1, slot_dim)))
            init.xavier_uniform_(self.slots_logsigma[i])
            self.to_q.append(nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False)))
            self.gru.append(nn.GRUCell(slot_dim, slot_dim))
            self.mlp.append(nn.Sequential(
                nn.LayerNorm(slot_dim),
                nn.Linear(slot_dim, mlp_hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_hidden_size, slot_dim)
            ))




    def forward(self, x):
        """
        input:
            feat: visual feature with position information, BxNxC
        output: slots: KxC, attn: KxN
        """

        slots = []
        for i in range(self.num_slots):
            mu = self.slots_mu[i].expand(1, -1)
            sigma = self.slots_logsigma[i].exp().expand(1, -1)
            slot = mu + sigma * torch.randn_like(mu)
            slots.append(slot)


        x = self.norm_inputs(x)

        k_input = self.to_k(x)
        v_input = self.to_v(x)

        for _ in range(self.num_iterations):
            slot_prev = slots

            attn_logits = []
            for i in range(self.num_slots):

                q_slots = self.to_q[i](slots[i])
                q_slots = q_slots*self.slot_dim ** -0.5

                cur_attn = torch.matmul(k_input, q_slots.T)
                attn_logits.append(cur_attn)

            attn_logits = torch.cat(attn_logits, dim=-1)
            attn = attn_logits.softmax(dim=-1)


            new_slots = []
            for i in range(self.num_slots):
                attn_slot = attn[...,i:i+1]
                weight = attn_slot + self.epsilon
                weight = weight / torch.sum(weight, dim=-2, keepdims=True)
                updates = torch.matmul(weight.T, v_input)

                new_slot = self.gru[i](
                    updates.reshape(-1, self.slot_dim),
                    slot_prev[i].reshape(-1, self.slot_dim)
                )

                new_slot = new_slot + self.mlp[i](new_slot)
                new_slots.append(new_slot)
            slots = new_slots


        

        slots = torch.cat(slots, dim=0)

        return slots, attn.T


class Decoder_nerf_ind(nn.Module):
    def __init__(self, device, position_emb_dim=5, input_dim=64, mlp_hidden_size=64):
        super().__init__()
        self.device = device

        # self.H, self.W, self.focal=cam_param
        self.position_emb_dim = position_emb_dim
        input_dim = input_dim + position_emb_dim*3*2 +3
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


        


    def forward(self, pts, slots):
        K, C = slots.shape
        N = pts.shape[1]


        slots = slots[:,None,:]
        slots =  slots.expand([-1, N, -1])

        pts = torch.reshape(pts, [-1,3])
        encoded_pts = embedding_fn(pts)
        out_dim = encoded_pts.shape[-1]
        encoded_pts = torch.reshape(encoded_pts, [K, N, out_dim])

        input = torch.cat([encoded_pts, slots], dim=-1)
        input = torch.reshape(input, [-1, out_dim + C])

        chunk = 1024*64
        output = []
        for i in range(0, input.shape[0], chunk):
            tmp = self.decoder_mlp_layer_0(input[i:i+chunk])
            tmp = self.decoder_mlp_layer_1(torch.cat([input[i:i+chunk],tmp], dim=-1))

            raw_density = self.decoder_density(tmp)
            latent = self.decoder_latent(tmp)  
            raw_color = self.decoder_color(latent)  
             
            output_c = torch.cat([raw_color,raw_density], dim=1)

            output.append(output_c)

        output = torch.cat(output, dim=0)

        all_raws = torch.reshape(output, [K, N, 4])

        raw_masks = F.relu(all_raws[:, :, -1:], True)  # KxNx1
        masks = raw_masks / (raw_masks.sum(dim=0) + 1e-5)  # KxNx1
        raw_rgb = (all_raws[:, :, :3].tanh() + 1) / 2

        raw_sigma = raw_masks

        unmasked_raws = torch.cat([raw_rgb, raw_sigma], dim=2)  # KxNx4
        masked_raws = unmasked_raws * masks
        raws = masked_raws.sum(dim=0)
        
        return raws, masked_raws, unmasked_raws, masks 




class Decoder_nerf_ind_v2(nn.Module):
    def __init__(self, device, position_emb_dim=5, input_dim=64, mlp_hidden_size=64):
        super().__init__()
        self.device = device

        # self.H, self.W, self.focal=cam_param
        self.position_emb_dim = position_emb_dim
        input_dim = input_dim + position_emb_dim*3*2 +3
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
            nn.Linear(mlp_hidden_size, 4), 
            nn.ReLU(True))


    def forward(self, input):

        tmp = self.decoder_mlp_layer_0(input)
        output_c = self.decoder_mlp_layer_1(torch.cat([input,tmp], dim=-1))


        
        return output_c


class Decoder_nerf(nn.Module):
    def __init__(self, device, position_emb_dim=5, input_dim=64, mlp_hidden_size=64):
        super().__init__()

        # self.H, self.W, self.focal=cam_param
        self.position_emb_dim = position_emb_dim

        input_dim = input_dim + position_emb_dim*3*2 +3


        self.mlp_hidden_size = mlp_hidden_size



        self.decoder_mlp_bg_layer_0 = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_size), 
            nn.ReLU(True),
            nn.Linear(mlp_hidden_size, mlp_hidden_size), 
            nn.ReLU(True),
            nn.Linear(mlp_hidden_size, mlp_hidden_size), 
            nn.ReLU(True))

        self.decoder_mlp_bg_layer_1 =  nn.Sequential(        
            nn.Linear(mlp_hidden_size+input_dim, mlp_hidden_size), 
            nn.ReLU(True),
            nn.Linear(mlp_hidden_size, mlp_hidden_size), 
            nn.ReLU(True),
            nn.Linear(mlp_hidden_size, 4), 
            nn.ReLU(True))


        self.decoder_mlp_fg_layer_0 = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_size), 
            nn.ReLU(True),
            nn.Linear(mlp_hidden_size, mlp_hidden_size), 
            nn.ReLU(True),
            nn.Linear(mlp_hidden_size, mlp_hidden_size), 
            nn.ReLU(True))

        self.decoder_mlp_fg_layer_1 =  nn.Sequential(        
            nn.Linear(mlp_hidden_size+input_dim, mlp_hidden_size), 
            nn.ReLU(True),
            nn.Linear(mlp_hidden_size, mlp_hidden_size), 
            nn.ReLU(True),
            nn.Linear(mlp_hidden_size, 4), 
            nn.ReLU(True))

    def forward(self, points_bg, points_fg, slots):
        '''
        inputs: points_bg: points sampled in background   Nx3
                points_fg: points sampled in objects  (K-1)xNx3
                slots: generated by slot attention model  KxC
        outputs: rgb colors and volume density of bg and fg
        '''

        K, C = slots.shape
        N = points_bg.shape[0]

        
        slots_bg = slots[0:1, None, :]  # 1xC
        slots_fg = slots[1:, None, :]  # (K-1)xC


        slots_bg =  slots_bg.expand([-1, N, -1])
        slots_fg =  slots_fg.expand([-1, N, -1])

        encoded_points_bg = embedding_fn(points_bg)
        out_dim = encoded_points_bg.shape[-1]
        # encoded_points_bg = torch.reshape(encoded_points_bg, [N, self.out_dim])
        encoded_points_bg = encoded_points_bg[None, ...]

        
        points_fg = torch.reshape(points_fg, [-1,3])
        encoded_points_fg = embedding_fn(points_fg)
        encoded_points_fg = torch.reshape(encoded_points_fg, [K-1, N, out_dim])
        

        input_bg = torch.cat([encoded_points_bg, slots_bg], dim=-1)
        input_bg = input_bg[0]
        input_fg = torch.cat([encoded_points_fg, slots_fg], dim=-1)
        input_fg = torch.reshape(input_fg, [-1, out_dim + C])


        tmp = self.decoder_mlp_bg_layer_0(input_bg)
        raw_bg = self.decoder_mlp_bg_layer_1(torch.cat([input_bg, tmp], dim=-1))
        raw_bg = raw_bg[None, ...]

        tmp = self.decoder_mlp_fg_layer_0(input_fg)
        raw_fg = self.decoder_mlp_fg_layer_1(torch.cat([input_fg,tmp], dim=-1))
        raw_fg = torch.reshape(raw_fg, [K-1, N, 4])


        all_raws = torch.cat([raw_bg, raw_fg], dim=0)  # KxNx4


        return all_raws



class Encoder_Decoder_nerf():
    def __init__(self, cam_param, num_slots = 3, num_iterations = 2, isTrain=True, lr=5e-4, vgg=False, separate_decoder=True, position_emb=True):


        self.num_slots = num_slots
        self.num_iterations = num_iterations

        self.cam_param = cam_param

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.vgg = vgg
        if vgg:
            self.encoder = Encoder_VGG(cam_param, self.device, position_emb=position_emb)
            in_dim = 256
            if position_emb:
                in_dim += 3
        else:
            self.encoder = Encoder(cam_param, self.device)
            in_dim = 64
        self.encoder.to(self.device)
        self.slot_attention = SlotAttention_Three(
            self.device,
            num_iterations=self.num_iterations,
            num_slots=self.num_slots,
            in_dim = in_dim,
            slot_dim=64,
            mlp_hidden_size=128)
        self.slot_attention.to(self.device)
        self.separate_decoder=separate_decoder
        if separate_decoder:
            self.decoder = []
            for s in range(num_slots):
                self.decoder.append(Decoder_nerf_ind())
                self.decoder[s].to(self.device)

        else:
            # self.decoder = Decoder_nerf(self.device)
            # self.decoder.to(self.device)
            self.decoder = Decoder_nerf_ind(self.device)
            self.decoder.to(self.device)

        
        self.k_o = 0.05

        self.isTrain = isTrain

        if self.isTrain:  # only defined during training time
            self.optimizer = optim.Adam(chain(
                self.slot_attention.parameters(), self.decoder.parameters()
            ), lr=lr)

    def forward(self, images, depth_maps, c2ws, isTrain=True):
        '''
        input:  images: images  BxHxWx3
                d: depth map BxHxW
                c2w: camera-to-world matrix Bx4x4
        '''
        # if training[0]==1:
        #     images, depth_maps, c2ws = images[0:1], depth_maps[0:1], c2ws[0:1]

        H, W, focal = self.cam_param

        images = images.to(self.device)
        depth_maps = depth_maps.to(self.device)
        c2ws = c2ws.to(self.device)
        x = self.encoder(images, depth_maps, c2ws) # (B,H',W',C)


        # check = np.random.randint(0,4)
        # input_images = images[check:check+1]
        # input_depth_maps = depth_maps[check:check+1]
        # input_c2ws = c2ws[check:check+1]
        # x = self.encoder(input_images, input_depth_maps, input_c2ws) # (B,H',W',C)

        
        x = x.permute([0,2,3,1])
        x = torch.reshape(x, [-1, x.shape[-1]])


        slots, attn = self.slot_attention(x) # (N_slots, slot_size)


        if isTrain==True:
            pts, z_vals, rays_d, select_inds = sampling_points(self.cam_param, c2ws, is_selection=True)
        else:
            pts, z_vals,rays_d = sampling_points(self.cam_param, c2ws)


        B,N,N_samples,_ = pts.shape

        # {

        # chunk = 128*128

        # pts = pts.reshape([-1,3])
        # emb_pts = embedding_fn(pts)
        # emb_pts = emb_pts[None,...]
        # emb_pts = emb_pts.expand([self.num_slots,-1,-1])

        # slots = slots[:,None,:]
        # num_p = emb_pts.shape[1]
        # slots = slots.expand([-1, num_p, -1])
        # input = torch.cat([emb_pts, slots], dim=-1)
        # input = input.reshape([-1, input.shape[-1]])

        # for i in range(0, input.shape[0],chunk):
        #     all_raw = self.decoder(input[i:i+chunk])
        #     if i == 0:
        #         all_raws = all_raw
        #     else:
        #         all_raws = torch.cat([all_raws, all_raw], dim=0)
        # all_raws = all_raws.reshape([self.num_slots, num_p, 4])


        # # pts_bg, pts_fg = preprocess_pts(pts, self.num_slots)
        # # for i in range(0, pts_bg.shape[0],chunk):
        # #     all_raw = self.decoder(pts_bg[i:i+chunk,...], pts_fg[:, i:i+chunk,...], slots)
        # #     if i == 0:
        # #         all_raws = all_raw
        # #     else:
        # #         all_raws = torch.cat([all_raws, all_raw], dim=1)

        # raw_masks = F.relu(all_raws[:, :, -1:], True)  # KxNx1

        # masks = raw_masks / (raw_masks.sum(dim=0) + 1e-5)  # KxNx1
        # raw_rgb = (all_raws[:, :, :3].tanh() + 1.) / 2.
        # raw_sigma = raw_masks

        # unmasked_raws = torch.cat([raw_rgb, raw_sigma], dim=2)  # KxNx4
        # masked_raws = unmasked_raws * masks
        # raws = masked_raws.sum(dim=0)
        

        # }



        # {
        pts = pts[None,...]
        pts = pts.expand([self.num_slots, -1, -1, -1, -1])

        pts = torch.reshape(pts, [self.num_slots, -1, 3])

        raws, masked_raws, unmasked_raws, masks = self.decoder(pts, slots)


        # }





        raws = raws.reshape([B,N,N_samples,4])
        masked_raws = masked_raws.reshape([self.num_slots,B,N,N_samples,4])
        unmasked_raws = unmasked_raws.reshape([self.num_slots,B,N,N_samples,4])

        raws = raws.reshape([-1,N_samples,4])
        z_vals = z_vals.reshape([-1,N_samples])
        rays_d = rays_d.reshape([-1,3])


        rgb = raw2outputs(raws, z_vals, rays_d)
        rgb = rgb.reshape([B,N,3])





        if isTrain==True:
            # select_inds_x, select_inds_y, N_selection = select_inds
            loss_img = torch.reshape(images, [B, -1, 3])
            loss_img = loss_img[:,select_inds,...]
            self.loss = L2_loss(rgb, loss_img)

            return 
            
            # masked_raws_sigma = masked_raws[..., -1:]
            # max_sigma,_ = masked_raws_sigma.permute([1,2,3,4,0]).max(dim=-1)
            # self.overlap_loss = torch.mean(raws[...,-1:]-max_sigma)
            
        # points sampling


        if isTrain==False:
            slot_masked_rgb = []
            for s in range(self.num_slots):
                slot_mask_raws = masked_raws[s]
                slot_mask_raws = slot_mask_raws.reshape([-1,N_samples,4])
                slot_masked_rgb.append(raw2outputs(slot_mask_raws, z_vals, rays_d).reshape([B,N,3]))

            slot_unmasked_rgb = []
            for s in range(self.num_slots):
                slot_unmask_raws = unmasked_raws[s]
                slot_unmask_raws = slot_unmask_raws.reshape([-1,N_samples,4])
                slot_unmasked_rgb.append(raw2outputs(slot_unmask_raws, z_vals, rays_d).reshape([B,N,3]))

            rgb = torch.reshape(rgb, [B, H, W, 3])
            for s in range(self.num_slots):
                slot_masked_rgb[s] = torch.reshape(slot_masked_rgb[s], [B, H, W, 3])

            for s in range(self.num_slots):
                slot_unmasked_rgb[s] = torch.reshape(slot_unmasked_rgb[s], [B, H, W, 3])
        


        return rgb, slot_masked_rgb, slot_unmasked_rgb

    def backward(self, iter):


        loss = self.loss 

        # if iter >= 1600000:
        #     k_o = max(1., (iter-1600000)/10000) * self.k_o
        #     loss = loss + k_o * self.overlap_loss
        loss.backward()
        

    def update_grad(self, images, depth_maps, c2ws, iter, img_id=None):

        self.forward(images, depth_maps, c2ws, img_id=img_id)


        self.optimizer.zero_grad()
        self.backward(iter)

        self.optimizer.step()

        return self.loss

    def save_weights(self, path, i):


        torch.save({
            'epoch': i,
            'encoder_state_dict': self.encoder.state_dict(),
            'slot_attention_state_dict': self.slot_attention.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
        }, os.path.join(path,'model_{:08d}.pt'.format(i)))



    def load_weights(self, path, i):
        if i==0:
            return
        checkpoint = torch.load(os.path.join(path,'model_{:08d}.pt'.format(i)))
        if self.vgg==False:
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.slot_attention.load_state_dict(checkpoint['slot_attention_state_dict'])
        if self.separate_decoder:
            for s in self.num_slots:
                self.decoder.load_state_dict(checkpoint['decoder_state_list'][s])
        else:
            self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])



class Decoder_nerf_attn(nn.Module):
    def __init__(self, device, position_emb_dim=5, input_dim=64, mlp_hidden_size=64):
        super().__init__()
        self.device = device

        # self.H, self.W, self.focal=cam_param
        self.position_emb_dim = position_emb_dim
        input_dim = input_dim + position_emb_dim*3*2 +3 + 1
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


        


    def forward(self, pts, slots, attns):
        K, C = slots.shape
        N = pts.shape[1]


        slots = slots[:,None,:]
        slots =  slots.expand([-1, N, -1])

        pts = torch.reshape(pts, [-1,3])
        encoded_pts = embedding_fn(pts)
        out_dim = encoded_pts.shape[-1]
        encoded_pts = torch.reshape(encoded_pts, [K, N, out_dim])

        input = torch.cat([encoded_pts, slots, attns], dim=-1)
        input = torch.reshape(input, [-1, out_dim + C + 1])

        chunk = 1024*64
        output = []
        for i in range(0, input.shape[0], chunk):
            tmp = self.decoder_mlp_layer_0(input[i:i+chunk])
            tmp = self.decoder_mlp_layer_1(torch.cat([input[i:i+chunk],tmp], dim=-1))

            raw_density = self.decoder_density(tmp)
            latent = self.decoder_latent(tmp)  
            raw_color = self.decoder_color(latent)  
             
            output_c = torch.cat([raw_color,raw_density], dim=1)

            output.append(output_c)

        output = torch.cat(output, dim=0)

        all_raws = torch.reshape(output, [K, N, 4])

        raw_masks = F.relu(all_raws[:, :, -1:], True)  # KxNx1
        masks = raw_masks / (raw_masks.sum(dim=0) + 1e-5)  # KxNx1
        raw_rgb = (all_raws[:, :, :3].tanh() + 1) / 2

        raw_sigma = raw_masks

        unmasked_raws = torch.cat([raw_rgb, raw_sigma], dim=2)  # KxNx4
        masked_raws = unmasked_raws * masks
        raws = masked_raws.sum(dim=0)
        
        return raws, masked_raws, unmasked_raws, masks 

class Encoder_Decoder_nerf_attn():
    def __init__(self, cam_param, num_slots = 3, num_iterations = 2, isTrain=True, lr=5e-4,
                    device = torch.device("cuda:0"),
                    num_images=0, c2ws_all=None, depth_all=None,
                    world_range_min=-10, world_range_max=10):


        self.num_slots = num_slots
        self.num_iterations = num_iterations

        self.cam_param = cam_param

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)


        self.encoder = Encoder_VGG(cam_param, self.device)
        in_dim = 256 + 3
        self.encoder.to(self.device)

        self.slot_attention = SlotAttention_Three(
            self.device,
            num_iterations=self.num_iterations,
            num_slots=self.num_slots,
            in_dim = in_dim,
            slot_dim=64,
            mlp_hidden_size=128)
        self.slot_attention.to(self.device)

        self.decoder = Decoder_nerf_attn(self.device)
        self.decoder.to(self.device)

        
        self.k_o = 0.05

        self.isTrain = isTrain

        if self.isTrain:  # only defined during training time
            self.optimizer = optim.Adam(chain(
                self.slot_attention.parameters(), self.decoder.parameters()
            ), lr=lr)
        
        if num_images > 0:
            h, w, f = self.cam_param
            self.attentions_2d = np.zeros((num_images, num_slots, h*w))
            rays_o, rays_d = get_rays(h, w, f, c2ws_all)
            self.surface_pts = get_position(rays_o, rays_d, depth_all)
            self.surface_pts = self.surface_pts.reshape([-1, 3])
            self.surface_pts.requires_grad = False
            self.surface_pts = self.surface_pts.cpu().numpy()
            self.world_range_min = world_range_min
            self.world_range_max = world_range_max

    def forward(self, images, depth_maps, c2ws, isTrain=True, img_id=None):
        '''
        input:  images: images  BxHxWx3
                d: depth map BxHxW
                c2w: camera-to-world matrix Bx4x4
        '''

        H, W, focal = self.cam_param

        images = images.to(self.device)
        depth_maps = depth_maps.to(self.device)
        c2ws = c2ws.to(self.device)
        x = self.encoder(images, depth_maps, c2ws) # (B,H',W',C)

        
        x = x.permute([0,2,3,1])
        x = torch.reshape(x, [-1, x.shape[-1]])


        slots, attn = self.slot_attention(x) # (N_slots, slot_size)


        if isTrain==True:
            pts, z_vals, rays_d, select_inds = sampling_points(self.cam_param, c2ws, is_selection=True)
        else:
            pts, z_vals,rays_d = sampling_points(self.cam_param, c2ws)
        
        B,N,N_samples,_ = pts.shape


        pts = pts[None,...]
        pts = pts.expand([self.num_slots, -1, -1, -1, -1])

        pts = torch.reshape(pts, [self.num_slots, -1, 3])

        if img_id is not None:
            if B == 1:
                if isTrain:
                    self.attentions_2d[img_id] = attn.detach().cpu().numpy()
            else:
                print('To be implemented, exit')
                exit()
            attn_3ds = []
            for k in range(self.num_slots):
                num_bins=64
                Hist, bin_edges = np.histogramdd(self.surface_pts, bins=num_bins,
                    range=[(self.world_range_min, self.world_range_max), 
                           (self.world_range_min, self.world_range_max), 
                           (self.world_range_min, self.world_range_max)],
                    weights=np.reshape(self.attentions_2d[:, k, :], (-1,)))
                x_id = np.searchsorted(bin_edges[0], pts[0, :, 0].detach().cpu().numpy())
                y_id = np.searchsorted(bin_edges[1], pts[0, :, 1].detach().cpu().numpy())
                z_id = np.searchsorted(bin_edges[2], pts[0, :, 2].detach().cpu().numpy())
                x_id = np.clip(x_id, 0, num_bins-1)
                y_id = np.clip(y_id, 0, num_bins-1)
                z_id = np.clip(z_id, 0, num_bins-1)
                attn_3d = Hist[x_id, y_id, z_id]
                attn_3ds.append(attn_3d)

            attn_3ds = np.array(attn_3ds)[..., None]
            attn_3ds = torch.from_numpy(attn_3ds).to(self.device).float()


        raws, masked_raws, unmasked_raws, masks = self.decoder(pts, slots, attn_3ds)


        raws = raws.reshape([B,N,N_samples,4])
        masked_raws = masked_raws.reshape([self.num_slots,B,N,N_samples,4])
        unmasked_raws = unmasked_raws.reshape([self.num_slots,B,N,N_samples,4])

        raws = raws.reshape([-1,N_samples,4])
        z_vals = z_vals.reshape([-1,N_samples])
        rays_d = rays_d.reshape([-1,3])


        rgb = raw2outputs(raws, z_vals, rays_d)
        rgb = rgb.reshape([B,N,3])





        if isTrain==True:
            # select_inds_x, select_inds_y, N_selection = select_inds
            loss_img = torch.reshape(images, [B, -1, 3])
            loss_img = loss_img[:,select_inds,...]
            self.loss = L2_loss(rgb, loss_img)

            return 
            
            # masked_raws_sigma = masked_raws[..., -1:]
            # max_sigma,_ = masked_raws_sigma.permute([1,2,3,4,0]).max(dim=-1)
            # self.overlap_loss = torch.mean(raws[...,-1:]-max_sigma)
            
        # points sampling


        if isTrain==False:
            slot_masked_rgb = []
            for s in range(self.num_slots):
                slot_mask_raws = masked_raws[s]
                slot_mask_raws = slot_mask_raws.reshape([-1,N_samples,4])
                slot_masked_rgb.append(raw2outputs(slot_mask_raws, z_vals, rays_d).reshape([B,N,3]))

            slot_unmasked_rgb = []
            for s in range(self.num_slots):
                slot_unmask_raws = unmasked_raws[s]
                slot_unmask_raws = slot_unmask_raws.reshape([-1,N_samples,4])
                slot_unmasked_rgb.append(raw2outputs(slot_unmask_raws, z_vals, rays_d).reshape([B,N,3]))

            rgb = torch.reshape(rgb, [B, H, W, 3])
            for s in range(self.num_slots):
                slot_masked_rgb[s] = torch.reshape(slot_masked_rgb[s], [B, H, W, 3])

            for s in range(self.num_slots):
                slot_unmasked_rgb[s] = torch.reshape(slot_unmasked_rgb[s], [B, H, W, 3])
        


        return rgb, slot_masked_rgb, slot_unmasked_rgb

    def backward(self, iter):


        loss = self.loss 

        # if iter >= 1600000:
        #     k_o = max(1., (iter-1600000)/10000) * self.k_o
        #     loss = loss + k_o * self.overlap_loss
        loss.backward()
        

    def update_grad(self, images, depth_maps, c2ws, iter, img_id=None):

        self.forward(images, depth_maps, c2ws, img_id=img_id)


        self.optimizer.zero_grad()
        self.backward(iter)

        self.optimizer.step()

        return self.loss

    def save_weights(self, path, i):


        torch.save({
            'epoch': i,
            'encoder_state_dict': self.encoder.state_dict(),
            'slot_attention_state_dict': self.slot_attention.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
        }, os.path.join(path,'model_{:08d}.pt'.format(i)))



    def load_weights(self, path, i):
        if i==0:
            return
        checkpoint = torch.load(os.path.join(path,'model_{:08d}.pt'.format(i)))
        if self.vgg==False:
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.slot_attention.load_state_dict(checkpoint['slot_attention_state_dict'])
        if self.separate_decoder:
            for s in self.num_slots:
                self.decoder.load_state_dict(checkpoint['decoder_state_list'][s])
        else:
            self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
