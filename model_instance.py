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
from helper_fun import *

import os


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

class SlotAttention_without_refine(nn.Module):

    def __init__(self, device, num_slots=3, in_dim=64+3, slot_dim=64, mlp_hidden_size=128, epsilon=1e-8):
        super().__init__()

        self.device = device
        self.num_slots = num_slots
        self.in_dim =in_dim
        self.slot_dim = slot_dim
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon

        self.norm_inputs = nn.LayerNorm(in_dim)
        self.to_k = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_v = nn.Linear(in_dim, slot_dim, bias=False)
        

        self.slots = nn.ParameterList()

        for i in range(num_slots):
            self.slots.append(nn.Parameter(torch.randn(1, slot_dim)))



    def forward(self, x):
        slots = []
        for i in range(self.num_slots):
            slots.append(self.slots[i].clone())
            slots[i] = slots[i].to(self.device)


        x = self.norm_inputs(x)

        # k_input = self.to_k(x)


        attn_logits = []
        for i in range(self.num_slots):
                    
            cur_attn = torch.matmul(x, slots[i].T)
            attn_logits.append(cur_attn)

        attn_logits = torch.cat(attn_logits, dim=-1)
        attn = attn_logits.softmax(dim=-1)

        slots = torch.cat(slots, dim=0)


        return slots, attn

class Decoder_NeRF_instanceSlot(nn.Module):
    def __init__(self, device, position_emb_dim=5, slot_input_dim=64, mlp_hidden_size=64):
        super().__init__()
        self.device = device

        # self.H, self.W, self.focal=cam_param
        self.position_emb_dim = position_emb_dim
        nerf_input_dim = slot_input_dim + position_emb_dim*3*2 +3
        self.mlp_hidden_size = mlp_hidden_size


        self.decoder_slot_layer = nn.Sequential(
            nn.Linear(slot_input_dim, slot_input_dim), 
            nn.ReLU(True),
            nn.Linear(slot_input_dim, slot_input_dim), 
            nn.ReLU(True),
            nn.Linear(slot_input_dim, slot_input_dim), 
            nn.ReLU(True))

        self.decoder_nerf_layer_0 = nn.Sequential(
            nn.Linear(nerf_input_dim, mlp_hidden_size), 
            nn.ReLU(True),
            nn.Linear(mlp_hidden_size, mlp_hidden_size), 
            nn.ReLU(True),
            nn.Linear(mlp_hidden_size, mlp_hidden_size), 
            nn.ReLU(True))

        self.decoder_nerf_layer_1 =  nn.Sequential(        
            nn.Linear(mlp_hidden_size+nerf_input_dim, mlp_hidden_size), 
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

        slots = self.decoder_slot_layer(slots)

        slots = slots[:,None,:]
        slots =  slots.expand([-1, N, -1])

        pts = torch.reshape(pts, [-1,3])
        encoded_pts = embedding_fn(pts)
        out_dim = encoded_pts.shape[-1]
        encoded_pts = torch.reshape(encoded_pts, [K, N, out_dim])

        input = torch.cat([encoded_pts, slots], dim=-1)
        input = torch.reshape(input, [-1, out_dim + C])

        # print(input.shape)
        # exit()

        chunk = 1024*64
        output = []
        for i in range(0, input.shape[0], chunk):
            tmp = self.decoder_nerf_layer_0(input[i:i+chunk])
            tmp = self.decoder_nerf_layer_1(torch.cat([input[i:i+chunk],tmp], dim=-1))

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



        self.slot_attention = SlotAttention_without_refine(
            self.device,
            in_dim = in_dim,
            slot_dim = in_dim)
        self.slot_attention.to(self.device)

        self.decoder = Decoder_NeRF_instanceSlot(self.device, slot_input_dim = in_dim)
        self.decoder.to(self.device)


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


        B, H, W, C = images.shape
        x = self.encoder(images, depth_maps, c2ws) # (B,H,W,C)

        
        x = x.permute([0,2,3,1])

        x = torch.reshape(x, [-1, x.shape[-1]])


        slots, attn = self.slot_attention(x) # (N_slots, slot_size)

        attn = attn.reshape([B, H, W, self.num_slots])


        if isTrain==True:
            pts, z_vals, rays_d, select_inds = sampling_points(self.cam_param, c2ws, is_selection=True)
        else:
            check = np.random.randint(0,4)
            pts, z_vals,rays_d = sampling_points(self.cam_param, c2ws[check:check+1])



        B,N,N_samples,_ = pts.shape

        pts = pts[None,...]
        pts = pts.expand([self.num_slots, -1, -1, -1, -1])
        pts = torch.reshape(pts, [self.num_slots, -1, 3])
        # raws, masked_raws, unmasked_raws, masks = self.decoder(pts, slots)

        chunk = 524288
        raws, masked_raws, unmasked_raws, masks = [], [], [], []
        for i in range(0, pts.shape[1], chunk):
            raws_c, masked_raws_c, unmasked_raws_c, masks_c = self.decoder(pts[:,i:i+chunk], slots)
            raws.append(raws_c)
            masked_raws.append(masked_raws_c)
            unmasked_raws.append(unmasked_raws_c)
            masks.append(masks_c)
            

        # }
        raws = torch.cat(raws, dim = 0)
        masked_raws = torch.cat(masked_raws, dim = 1)
        unmasked_raws = torch.cat(unmasked_raws, dim = 1)
        masks = torch.cat(masks, dim = 1)



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
            
            
            slot_masked_rgb = []
            for s in range(self.num_slots):
                slot_mask_raws = masked_raws[s]
                slot_mask_raws = slot_mask_raws.reshape([-1,N_samples,4])
                slot_masked_rgb.append(raw2outputs(slot_mask_raws, z_vals, rays_d).reshape([B,N,3]))
            slot_masked_rgb = torch.stack(slot_masked_rgb, dim=0)

            loss_attn = torch.reshape(attn, [B, -1, self.num_slots])
            loss_attn = loss_attn[:,select_inds,...]
            loss_attn = loss_attn.permute([2,0,1])
            loss_attn = loss_attn[...,None]
            
            loss_masked_img = loss_img[None,...]
            loss_masked_img = loss_masked_img.expand([3,-1,-1,-1])
            loss_masked_img = loss_masked_img * loss_attn
            


            self.loss = L2_loss(rgb, loss_img) + L2_loss(slot_masked_rgb, loss_masked_img)

            return 
            
            # masked_raws_sigma = masked_raws[..., -1:]
            # max_sigma,_ = masked_raws_sigma.permute([1,2,3,4,0]).max(dim=-1)
            # self.overlap_loss = torch.mean(raws[...,-1:]-max_sigma)
            
        # points sampling


        else:
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
        

    def update_grad(self, images, depth_maps, c2ws, iter):

        self.forward(images, depth_maps, c2ws)


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





