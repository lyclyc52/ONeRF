import os
import torch
import torchvision.transforms.functional as TF
import numpy as np
import imageio 
import json

def load_blender_data(basedir, half_res=False, testskip=1, size=-1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]

    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    if basedir=='./data/nerf_synthetic/clevr_100_2objects':
        focal = 875.
    
    render_poses = None
    
    if size > 0:
        imgs = torch.from_numpy(imgs).permute([0,3,1,2])[:, :3, ...]
        imgs = TF.resize(imgs, size=size)
        H = H * size//800
        W = W * size//800
        focal = focal * size/800.
    elif half_res:
        imgs = torch.from_numpy(imgs).permute([0,3,1,2])[:, :3, ...]
        imgs = TF.resize(imgs, size=400)
        H = H * 128//800
        W = W * 128//800
        focal = focal * 128/800.
        
    return imgs, poses, render_poses, [H, W, focal], i_split


