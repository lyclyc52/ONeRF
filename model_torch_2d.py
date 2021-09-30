import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg16
from torchvision.transforms import Normalize
from torch.nn import init

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

def build_grid(resolution):
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges)
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], -1])
    grid = grid.unsqueeze(0)
    return torch.cat([grid, 1.0 - grid], dim=-1)

class SoftPositionEmbed(nn.Module):
    def __init__(self, out_channels, in_size):
        super().__init__()
        self.dense = nn.Linear(in_features=4, out_features=out_channels)
        self.register_buffer("grid", build_grid(in_size))

    def forward(self, inputs):
        emb_proj = self.dense(self.grid).permute(0, 3, 1, 2)
        return inputs + emb_proj

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
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), 
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

class Decoder2d(nn.Module):
    def __init__(self, position_emb=True,
            decoder_initial_size=(8, 8)):
        
        super().__init__()

        self.position_emb = position_emb
        self.decoder_initial_size = decoder_initial_size
        if position_emb:
            self.decoder_pos = SoftPositionEmbed(
                out_channels=64, 
                in_size=self.decoder_initial_size)

        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 5, stride=1, padding=2, output_padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 4, 3, stride=1, padding=1, output_padding=0),
        )

    def forward(self, x):
        if self.position_emb:
            x = self.decoder_pos(x)
        x = self.decoder_cnn(x)

        return x #(B, H, W, 4)

class EncoderDecoder2d(nn.Module):
    def __init__(self, cam_param, num_slots = 3, num_iterations = 2,
            decoder_initial_size = (8, 8)):
        super().__init__()
        self.cam_param = cam_param
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.encoder = Encoder_VGG(cam_param)
        self.decoder_initial_size = decoder_initial_size
        self.slot_attention = SlotAttention(
            num_iterations=self.num_iterations,
            num_slots=self.num_slots,
            in_dim = 256 + 3,
            slot_dim=64,
            mlp_hidden_size=128)
        self.decoder = Decoder2d(decoder_initial_size=self.decoder_initial_size)

    def forward(self, images, depth_maps, c2ws):
        device = images.device
        H, W, focal = self.cam_param
        x = self.encoder(images, depth_maps, c2ws)

        x = x.permute([0,2,3,1])
        B, H, W, C = x.shape
        x = torch.reshape(x, [-1, x.shape[-1]])


        slots, attn = self.slot_attention(x) # (N_slots, slot_size)

        slots = slots[..., None, None]
        decoder_in = slots.repeat(1, 1, 
                self.decoder_initial_size[0], self.decoder_initial_size[1])
        out = self.decoder(decoder_in)

        out = out.permute([0, 2, 3, 1])
        recons = out[..., :3]
        masks = out[..., :-1]
        masks = F.softmax(masks, dim=0)
        recon_combined = torch.sum(recons * masks, dim=0)
        
        return recon_combined, recons, masks, slots
