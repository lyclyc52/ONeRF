import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"]='5,6'
from model_clustering import *
from load_blender import *
from run_nerf_helpers import *
from run_nerf import *
import torch



def load_data(basedir, half_res=False, testskip=1, size=-1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    all_depths = []
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

    for i in i_split[0]:
        dname = os.path.join(basedir, 'depth', 'depth_{}.png'.format(i) )
        all_depths.append(imageio.imread(fname))

    all_depths = (np.array(all_depths) / 255. * 30).astype(np.float32)

    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]

    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    # if basedir=='./data/nerf_synthetic/clevr_100_2objects':
    focal = 875.
    
    render_poses = tf.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]],0)

    if size > 0:
        imgs = tf.compat.v1.image.resize_area(imgs, [size, size]).numpy()
        all_depths = tf.compat.v1.image.resize_area(all_depths, [size, size]).numpy()
        H = H * size//H
        W = W * size//W
        focal = focal * size/ 400.

    all_depths = all_depths[..., 0]
        
    return imgs, poses, all_depths, render_poses, [H, W, focal], i_split




weights_dir = './results/testing_3/weights'
img_dir = './results/testing_10/imgs'

input_size = 128

os.makedirs(weights_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)



datadir = 'data/nerf_synthetic/clevr_depth'


images, poses, depth_maps, render_poses, hwf, i_split = load_data(
            datadir, True, 1, size = input_size)


N_imgs=49
images, depth_maps, poses = images[:N_imgs, :, :, :3], depth_maps[:N_imgs], poses[:N_imgs]
images, depth_maps, poses = torch.from_numpy(images), torch.from_numpy(depth_maps), torch.from_numpy(poses)
loss_fn = torch.nn.CrossEntropyLoss()


loss_hpy = torch.nn.L1Loss(size_average = True)
loss_hpz = torch.nn.L1Loss(size_average = True)

device = torch.device("cuda:0" )
f_extractor = Encoder_VGG(hwf, device=device)
f_extractor.to(device)

C = 259
num_slots = 2 
slots = torch.randn(num_slots, C-3)
position = torch.randn(num_slots, 3)
slots = slots.to(device)
position = position.to(device)


val = [0, 2, 22, 39, 41]



mask_dir = 'results/testing_9/imgs'
masks = []
for i in range(5):
    mask_f = os.path.join(mask_dir,'val_{:06d}_slot1.jpg'.format(i))
    masks.append(imageio.imread(mask_f))

masks = ( np.array(masks) / 255.).astype(np.float32)

masks = torch.from_numpy(masks)
masks = masks.to(device)

print(masks.shape)

with torch.no_grad():
    for iter in range(1):
        # val = [0, 3, 25, 39]
        # val = [t for t in range(iter*2,iter*2+4)]
        # val = [46, 47, 48]

        val = [0, 2, 22, 39, 41]
        print(val)
        val_images, val_depths, val_poses = images[val], depth_maps[val], poses[val]

        # val_images = val_images * masks[...,None]


        val_images, val_depths, val_poses = val_images.to(device), val_depths.to(device), val_poses.to(device)



        val_depths = val_depths * masks + (1. - masks) * 1e10
        print(val_images.shape)
        print(val_depths.shape)
        f = f_extractor(val_images, val_depths, val_poses)

        B, H, W, C = f.shape

        masks = masks.reshape(-1)
        mask_index = masks>0.5

        f = f.reshape([-1, C])
        f = f[mask_index]






        # normal_f = nn.LayerNorm(C-3)
        # normal_f.to(device)
        # f = normal_f(f)



        w = 100.
        # f[...,C-3:] = w * f[...,C-3:]
        # with torch.no_grad():
        #     for i in range(20):
        #         z = torch.matmul(f, slots)      # NxK
        #         # z_p = torch.matmul(f_p, position) 
        #         # z = z + w * z_p
        #         z = F.softmax(z, dim=-1)                 # NxK
        #         z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
        #         slots = torch.matmul(f.T, z_)       # CxK
        #         slots = _l2norm(slots, dim=0)


        # attn_logits = torch.matmul(f, slots)
        # attn = attn_logits.softmax(dim=-1)
        # print(attn.shape)
        # attn = attn.reshape([B,H,W,num_slots])
        # attn = attn.permute([0,3,1,2])



        f_p = f[...,C-3:]
        f = f[...,:C-3]

        normal_f = nn.LayerNorm(C-3)
        normal_f.to(device)
        f = normal_f(f)

        for i in range(1):
            z=[]
            z_p=[]
            for j in range(num_slots):
                z.append(torch.sum((f - slots[j])**2, dim=-1))    # N
                # if j!= num_slots-1:
                #     z_p.append(torch.sum((f_p - position[:, j])**2, dim=-1))
                z_p.append(torch.sum((f_p - position[j])**2, dim=-1)) # N
            z = torch.stack(z)
            z_p = torch.stack(z_p)
            z = z.T
            z_p = z_p.T 


            # print(z_p.shape)
            # print(z.shape)

            # bg_p = torch.zeros([z.shape[0],1])
            # bg_p = bg_p.to(device)
            # z_p = torch.cat([z_p,bg_p],dim=1)

            score = z+w*z_p
            ignore, index = torch.min(score,1)
            
            for j in range(num_slots):
                slots[j] = torch.mean(f[index==j], dim=0)
                # if j!= num_slots-1:
                #     position[:, j]  = torch.mean(f_p[index==j].T, dim=-1)
                position[j] = torch.mean(f_p[index==j], dim=0)


        sub_attn_logits = score
        sub_attn = sub_attn_logits.softmax(dim=-1)
        print(sub_attn.shape)


        attn0 = masks.clone()
        attn1 = masks.clone()

        attn0[mask_index] = sub_attn[...,0]
        attn1[mask_index] = sub_attn[...,1]
        attn0 = attn0.reshape([B,H,W])
        attn1 = attn1.reshape([B,H,W])
        attn0 = attn0[...,None]
        attn1 = attn1[...,None]


        attn = torch.cat([attn0, attn1], dim = -1)


        attn = attn.permute([0,3,1,2])

        masks = masks.reshape([B,H,W])

        attn = attn.cpu().numpy()
        val_images = val_images.cpu().numpy()
        masks = masks.cpu().numpy()
        print(iter)
        print(B)
        for b in range(B):
            for s in range(num_slots):
                
                imageio.imwrite(os.path.join(img_dir, 'val_{:06d}_slot{:01d}.jpg'.format(b,s)), to8b(attn[b][s]))
                imageio.imwrite(os.path.join(img_dir, 'masked_{:06d}_slot{:01d}.jpg'.format(b,s)), to8b(attn[b][s][...,None]*val_images[b]))
            # imageio.imwrite(os.path.join(img_dir, 'gt_{:06d}.jpg'.format(b)), to8b(val_images[b]))
            # imageio.imwrite(os.path.join(img_dir, 'mask_{:06d}.jpg'.format(b)), to8b(masks[b,...,None]))