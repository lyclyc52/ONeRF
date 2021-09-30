import torch
import numpy as np


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



def sampling_points(cam_param, c2w, N_samples=64, near=4., far=14., is_selection=False, N_selection=64*16):
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
